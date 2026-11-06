import argparse
import csv
import os
import re
import time
import traceback
import torch
import torchaudio
import torchvision
from datamodule.transforms import VideoTransform
from infer_module import InferenceModule


_CHARS_TO_IGNORE = r"[\,\?\.\!\-\;\:\"]"
def normalize_text(text):
    text = re.sub(_CHARS_TO_IGNORE, "", text)
    text = text.upper().replace("\u2019", "'")
    return " ".join(text.split())


def wer_distance(ref, hyp):
    return torchaudio.functional.edit_distance(
        ref.lower().split(),
        hyp.lower().split(),
    )


def read_test_list(test_list_path):
    if not os.path.isfile(test_list_path):
        raise FileNotFoundError(f"Test list not found: {test_list_path}")
    with open(test_list_path, encoding="utf-8") as f:
        clips = [line.strip().split()[0] for line in f if line.strip()]
    return clips


def read_ground_truth(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        if line.startswith("Text:"):
            return line[len("Text:"):].strip()
    return content.strip()


def load_preprocessed_clip(mp4_path, vtx, device, half):
    video_np = torchvision.io.read_video(mp4_path, pts_unit="sec")[0].numpy()
    if video_np.shape[0] == 0:
        return None
    video_tensor = torch.tensor(video_np).permute(0, 3, 1, 2)
    video_tensor = vtx(video_tensor)
    if half:
        video_tensor = video_tensor.half()
    return video_tensor.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-root", required=True)
    ap.add_argument("--test-list", default=None,
                    help="LRS2-style split file. Use this for the official test set.")
    ap.add_argument("--val-from-labels", default=None,
                    help="A make_labels.py output file. Reference transcripts come "
                         "from the labels file rather than the .txt next to the video. "
                         "Useful for val sweeps because we only have main/ test files "
                         "as .txt sidecars.")
    ap.add_argument("--ckpt", required=True, help="Visual model checkpoint")
    ap.add_argument("--lm-ckpt", default=None,
                    help="Optional Transformer LM checkpoint (.pt from train_lm.py)")
    ap.add_argument("--lm-weight", type=float, default=0.0)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--length-bonus", type=float, default=0.0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--out-csv", default="lrs2_results.csv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.test_list is None and args.val_from_labels is None:
        ap.error("Must specify either --test-list or --val-from-labels.")
    if args.test_list is not None and args.val_from_labels is not None:
        ap.error("Specify only ONE of --test-list / --val-from-labels.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[eval] Device: {device}")
    print(f"[eval] Visual ckpt: {args.ckpt}")
    if args.lm_ckpt:
        print(f"[eval] LM ckpt: {args.lm_ckpt}")
        print(f"[eval] LM weight: {args.lm_weight}, ctc_weight: {args.ctc_weight}, "
              f"length_bonus: {args.length_bonus}")
    else:
        print(f"[eval] No LM (lm_weight={args.lm_weight}). Shallow fusion disabled.")

    module = InferenceModule(
        modality="video",
        ckpt_path=args.ckpt,
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        lm_ckpt_path=args.lm_ckpt,
        lm_weight=args.lm_weight,
        penalty=args.length_bonus,
    )
    module.to(device).eval()

    vtx = VideoTransform(subset="test")

    transcript_from_labels = None
    if args.val_from_labels is not None:
        clips = []
        transcript_from_labels = {}
        with open(args.val_from_labels, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    clip_id = parts[0]
                    transcript = parts[2].rstrip("\r")
                    clips.append(clip_id)
                    transcript_from_labels[clip_id] = transcript
        print(f"[eval] {len(clips)} clips from {args.val_from_labels}")
    else:
        clips = read_test_list(args.test_list)
        print(f"[eval] {len(clips)} clips from {args.test_list}")

    if args.max_clips is not None:
        clips = clips[: args.max_clips]
        print(f"[eval] limited to first {len(clips)} clips by --max-clips")

    already_done = set()
    if args.resume and os.path.isfile(args.out_csv):
        with open(args.out_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(row["clip_id"])
        print(f"[eval] Resume: {len(already_done)} clips already in {args.out_csv}")

    new_file = not (args.resume and os.path.isfile(args.out_csv))
    mode = "w" if new_file else "a"
    fout = open(args.out_csv, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(
        fout,
        fieldnames=["clip_id", "reference", "hypothesis", "edit_distance", "ref_words", "status"],
    )
    if new_file:
        writer.writeheader()

    total_ed = 0
    total_ref_words = 0
    n_ok = 0
    n_skip = 0
    n_fail = 0
    t_start = time.time()

    for i, clip_id in enumerate(clips, 1):
        if clip_id in already_done:
            continue

        mp4 = os.path.join(args.preprocessed_root, f"{clip_id}.mp4")
        if transcript_from_labels is not None:
            ref_raw = transcript_from_labels.get(clip_id)
            if ref_raw is None:
                writer.writerow(dict(
                    clip_id=clip_id, reference="", hypothesis="",
                    edit_distance=0, ref_words=0, status="missing_label",
                ))
                n_skip += 1
                fout.flush()
                continue
        else:
            txt = os.path.join(args.preprocessed_root, f"{clip_id}.txt")
            if not os.path.isfile(txt):
                writer.writerow(dict(
                    clip_id=clip_id, reference="", hypothesis="",
                    edit_distance=0, ref_words=0, status="txt_missing",
                ))
                n_skip += 1
                fout.flush()
                continue
            ref_raw = read_ground_truth(txt)

        if not os.path.isfile(mp4):
            writer.writerow(dict(
                clip_id=clip_id, reference="", hypothesis="",
                edit_distance=0, ref_words=0, status="mp4_missing",
            ))
            n_skip += 1
            fout.flush()
            continue

        try:
            ref = normalize_text(ref_raw)
            video_tensor = load_preprocessed_clip(mp4, vtx, device, half=False)
            if video_tensor is None:
                writer.writerow(dict(
                    clip_id=clip_id, reference=ref, hypothesis="",
                    edit_distance=0, ref_words=len(ref.split()), status="empty_video",
                ))
                n_skip += 1
                fout.flush()
                continue
            hyp = module(video_tensor)
            hyp_norm = normalize_text(hyp)
            ed = wer_distance(ref, hyp_norm)
            ref_n = len(ref.split())
            total_ed += ed
            total_ref_words += ref_n
            n_ok += 1
            writer.writerow(dict(
                clip_id=clip_id, reference=ref, hypothesis=hyp_norm,
                edit_distance=ed, ref_words=ref_n, status="ok",
            ))
        except Exception as e:
            n_fail += 1
            writer.writerow(dict(
                clip_id=clip_id, reference="", hypothesis="",
                edit_distance=0, ref_words=0,
                status=f"error:{type(e).__name__}:{str(e)[:100]}",
            ))
            traceback.print_exc()

        fout.flush()

        if i % 10 == 0 or i == len(clips):
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1e-6)
            eta_s = (len(clips) - i) / max(rate, 1e-6)
            running_wer = (total_ed / total_ref_words * 100) if total_ref_words else float("nan")
            print(
                f"[eval] {i}/{len(clips)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                f"running WER={running_wer:.2f}%  rate={rate:.2f} clip/s  eta={eta_s/60:.1f}min",
                flush=True,
            )

    fout.close()

    print("\n" + "=" * 60)
    print(f"Evaluated: {n_ok} clips")
    print(f"Skipped:   {n_skip}")
    print(f"Failed:    {n_fail}")
    if total_ref_words > 0:
        wer = total_ed / total_ref_words
        print(f"Corpus WER = {total_ed} / {total_ref_words} = {wer:.4f} ({wer*100:.2f}%)")
    else:
        print("No reference words — cannot compute WER.")
    print(f"Per-clip results: {args.out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
