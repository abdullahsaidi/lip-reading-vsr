import argparse
import json
import os
import sys
import time
import warnings
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamodule.transforms import VideoTransform
from infer_module import InferenceModule
from eval_lrs2_preprocessed import (
    load_preprocessed_clip,
    normalize_text,
    read_test_list,
    read_ground_truth,
)

warnings.filterwarnings("ignore", category=UserWarning)


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
    ap.add_argument("--preprocessed-root", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels-file",
                     help="TSV: <rel_path>\\t<n_frames>\\t<transcript>")
    src.add_argument("--test-list",
                     help="LRS2 test list. Transcripts read from <root>/<id>.txt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam-size", type=int, default=50)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--max-clips", type=int, default=-1)
    ap.add_argument("--output-jsonl", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[nbest] Device: {device}", flush=True)
    print(f"[nbest] Visual ckpt: {args.ckpt}", flush=True)
    print(f"[nbest] beam_size: {args.beam_size}, ctc_weight: {args.ctc_weight}", flush=True)
    print(f"[nbest] Output: {args.output_jsonl}", flush=True)

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
    text_transform = module.text_transform
    eos_id = module.model.odim - 1
    vtx = VideoTransform(subset="test")

    # Build (clip_id, transcript) list
    clips = []
    if args.labels_file:
        print(f"[nbest] using TSV labels: {args.labels_file}")
        with open(args.labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line: continue
                parts = line.split("\t")
                if len(parts) < 3: continue
                clips.append((parts[0], parts[2].rstrip("\r")))
    else:
        print(f"[nbest] using test list + .txt sidecars: {args.test_list}")
        clip_ids = read_test_list(args.test_list)
        for cid in clip_ids:
            txt_path = os.path.join(args.preprocessed_root, f"{cid}.txt")
            if not os.path.isfile(txt_path):
                continue
            transcript = read_ground_truth(txt_path)
            clips.append((cid, transcript))

    if args.max_clips > 0:
        clips = clips[:args.max_clips]
    print(f"[nbest] {len(clips)} clips loaded", flush=True)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    out_f = open(args.output_jsonl, "w", encoding="utf-8")

    n_ok = n_skip = n_fail = 0
    total_ref_words = 0
    total_1best_errors = 0
    total_oracle_errors = 0

    t0 = time.time()
    for i, (clip_id, transcript) in enumerate(clips):
        video_path = os.path.join(args.preprocessed_root, clip_id + ".mp4")
        if not os.path.exists(video_path):
            n_skip += 1
            continue
        ref = normalize_text(transcript)
        ref_words = ref.split()

        try:
            video_tensor = load_preprocessed_clip(video_path, vtx, device, half=False)
            if video_tensor is None:
                n_skip += 1
                continue
            with torch.no_grad():
                x = module.model.frontend(video_tensor.unsqueeze(0))
                x = module.model.proj_encoder(x)
                enc_feat, _ = module.model.encoder(x, None)
                enc_feat = enc_feat.squeeze(0)
                nbest_hyps = module.beam_search(enc_feat)
        except Exception as e:
            print(f"[nbest] FAIL on {clip_id}: {e}", flush=True)
            import traceback; traceback.print_exc()
            n_fail += 1
            continue

        hyps = []
        oracle_errors_this = len(ref_words)
        for rank, hyp in enumerate(nbest_hyps):
            d = hyp.asdict()
            yseq = d["yseq"]
            tok_ids = [int(t) for t in yseq[1:]]
            if tok_ids and tok_ids[-1] == eos_id:
                tok_ids = tok_ids[:-1]
            tok_tensor = torch.tensor(tok_ids, dtype=torch.long)
            text = text_transform.post_process(tok_tensor).replace("<eos>", "")
            text = normalize_text(text)
            visual_score = float(d["score"])
            hyps.append({
                "rank": rank + 1,
                "text": text,
                "visual_score": visual_score,
                "n_tokens": len(tok_ids),
            })
            errors = edit_distance(ref_words, text.split())
            oracle_errors_this = min(oracle_errors_this, errors)

        n_ok += 1
        total_ref_words += len(ref_words)
        best_hyp_words = hyps[0]["text"].split() if hyps else []
        total_1best_errors += edit_distance(ref_words, best_hyp_words)
        total_oracle_errors += oracle_errors_this

        out_f.write(json.dumps({
            "clip_id": clip_id, "ref": ref, "hyps": hyps,
        }) + "\n")
        out_f.flush()

        if (i+1) % 10 == 0 or i+1 == len(clips):
            elapsed = time.time() - t0
            rate = (i+1) / max(elapsed, 1e-9)
            eta = (len(clips) - i - 1) / max(rate, 1e-9) / 60
            wer1 = total_1best_errors / max(total_ref_words, 1)
            wer_o = total_oracle_errors / max(total_ref_words, 1)
            print(f"[nbest] {i+1}/{len(clips)}  ok={n_ok} skip={n_skip} fail={n_fail}  "
                  f"1best_WER={wer1*100:.2f}%  oracle_WER={wer_o*100:.2f}%  "
                  f"rate={rate:.2f} clip/s  eta={eta:.1f}min", flush=True)

    out_f.close()
    print()
    print("=" * 60)
    print(f"DONE: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    if total_ref_words > 0:
        wer1 = total_1best_errors / total_ref_words * 100
        wer_o = total_oracle_errors / total_ref_words * 100
        print(f"1-best  WER: {total_1best_errors}/{total_ref_words} = {wer1:.2f}%")
        print(f"Oracle  WER: {total_oracle_errors}/{total_ref_words} = {wer_o:.2f}%")
        print(f"Headroom for rescoring: {wer1 - wer_o:.2f} WER points")
    print(f"Output: {args.output_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
