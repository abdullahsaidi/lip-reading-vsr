#Runs the visual model with beam search (NO LM rescoring) on all clips in the labels file and writes per-clip results plus a final WER.
import argparse
import csv
import os
import time
import torch
from infer_module import InferenceModule
from datamodule.transforms import VideoTransform
from eval_lrs2_preprocessed import load_preprocessed_clip, normalize_text


def edit_distance(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    ap.add_argument("--labels", required=True, help="path to labels file (tab-separated)")
    ap.add_argument("--root", required=True, help="preprocessed root, e.g. /path/to/main_mouth96/test")
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--limit", type=int, default=-1, help="if >0, evaluate only first N clips")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    print(f"[eval] loading {args.ckpt} ...")
    module = InferenceModule(
        modality="video",
        ckpt_path=args.ckpt,
        beam_size=args.beam_size,
        ctc_weight=args.ctc_weight,
        lm_ckpt_path=None,
        lm_weight=0.0,
        penalty=0.0,
    )
    module.to(device).eval()
    vtx = VideoTransform(subset="test")

    entries = []
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            clip_id, n_frames, transcript = parts
            entries.append((clip_id, int(n_frames), transcript.rstrip("\r")))
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"[eval] {len(entries)} clips, beam_size={args.beam_size}")

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    f_out = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_out)
    writer.writerow(["clip_id", "ref", "hyp", "n_words", "errors", "per_clip_wer"])

    total_errors = 0
    total_words = 0
    n_skip = 0
    t0 = time.time()

    for i, (clip_id, n_frames, transcript) in enumerate(entries):
        mp4 = os.path.join(args.root, f"{clip_id}.mp4")
        if not os.path.isfile(mp4):
            n_skip += 1
            continue
        try:
            video = load_preprocessed_clip(mp4, vtx, device, half=False)
            pred = module(video)
            pred = " ".join(pred.upper().split())
        except Exception as e:
            print(f"[eval] skip {clip_id}: {e}")
            n_skip += 1
            continue

        ref = normalize_text(transcript)
        ref_words = ref.split()
        hyp_words = pred.split()
        errs = edit_distance(ref_words, hyp_words)
        total_errors += errs
        total_words += len(ref_words)
        per_clip = errs / max(len(ref_words), 1)
        writer.writerow([clip_id, ref, pred, len(ref_words), errs, f"{per_clip:.4f}"])
        f_out.flush()

        if (i+1) % 50 == 0 or (i+1) == len(entries):
            elapsed = time.time() - t0
            wer = total_errors / max(total_words, 1)
            rate = (i+1) / elapsed
            remaining = len(entries) - i - 1
            eta_min = remaining / max(rate, 0.01) / 60
            print(f"[eval] {i+1}/{len(entries)}  running WER={wer*100:.2f}%  "
                  f"rate={rate:.2f}clips/s  eta={eta_min:.1f}min")

    f_out.close()

    n_ok = len(entries) - n_skip
    final_wer = total_errors / max(total_words, 1)
    print()
    print("=" * 70)
    print(f"FINAL WER")
    print(f"  checkpoint: {args.ckpt}")
    print(f"  labels:     {args.labels}")
    print(f"  clips:      {n_ok} evaluated, {n_skip} skipped")
    print(f"  errors:     {total_errors}")
    print(f"  words:      {total_words}")
    print(f"  WER:        {final_wer*100:.2f}%")
    print(f"  csv:        {args.out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
