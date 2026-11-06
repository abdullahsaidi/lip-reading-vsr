"""produce per-clip CSV comparing baseline 1-best vs Qwen-rescored.

For each test clip, output:
  clip_id, ref, n_ref_words,
  baseline_text, baseline_errors,
  qwen7b_text, qwen7b_rank_picked, qwen7b_errors, qwen7b_outcome,
  qwen14b_text, qwen14b_rank_picked, qwen14b_errors, qwen14b_outcome

outcome: "rescued" if rescoring picked a better hyp than rank 1
         "hurt"    if rescoring picked a worse hyp
         "same"    if rescoring picked rank 1 OR a tied-error hyp
         "neutral" if both 1-best and rescored are wrong by same amount but different"""

import argparse
import csv
import json
import os
import sys


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]


def pick_with_alpha_beta(hyps, llm_scores, alpha, beta):
    #Return (best_hyp, rank_picked) under alpha*lm + beta*length scoring.
    best_score = -float("inf")
    best_h = None
    best_rank = -1
    for i, (h, llm) in enumerate(zip(hyps, llm_scores)):
        n_words = len(h["text"].split())
        score = h["visual_score"] + alpha * llm + beta * n_words
        if score > best_score:
            best_score = score
            best_h = h
            best_rank = h["rank"]
    return best_h, best_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest-jsonl", default="results/test_beam25.jsonl")
    ap.add_argument("--qwen7b-cache", default="results/test_qwen7b_scores.jsonl")
    ap.add_argument("--qwen14b-cache", default="results/test_qwen14b_scores.jsonl")
    ap.add_argument("--qwen7b-alpha", type=float, default=0.2)
    ap.add_argument("--qwen7b-beta", type=float, default=0.2)
    ap.add_argument("--qwen14b-alpha", type=float, default=0.15)
    ap.add_argument("--qwen14b-beta", type=float, default=0.0)
    ap.add_argument("--out-csv", default="results/error_analysis_test.csv")
    args = ap.parse_args()

    print(f"Loading n-best: {args.nbest_jsonl}")
    clips = [json.loads(l) for l in open(args.nbest_jsonl)]
    print(f"Loaded {len(clips)} clips")
    print(f"Loading Qwen-7B cache: {args.qwen7b_cache}")
    cache_7b = {}
    for l in open(args.qwen7b_cache):
        r = json.loads(l)
        cache_7b[r["clip_id"]] = r["llm_scores"]

    print(f"Loading Qwen-14B cache: {args.qwen14b_cache}")
    cache_14b = {}
    for l in open(args.qwen14b_cache):
        r = json.loads(l)
        cache_14b[r["clip_id"]] = r["llm_scores"]

    print(f"Using 7B (alpha={args.qwen7b_alpha}, beta={args.qwen7b_beta})  "
          f"14B (alpha={args.qwen14b_alpha}, beta={args.qwen14b_beta})")
    print()

    # Classify outcomes per LM
    counts_7b = {"rescued": 0, "hurt": 0, "same": 0, "neutral": 0}
    counts_14b = {"rescued": 0, "hurt": 0, "same": 0, "neutral": 0}

    rows = []
    n_skip = 0
    for clip in clips:
        cid = clip["clip_id"]
        ref = clip["ref"]
        ref_words = ref.split()
        hyps = clip["hyps"]

        # Baseline = rank 1
        baseline_text = hyps[0]["text"]
        baseline_errors = edit_distance(ref_words, baseline_text.split())

        # 7B picked
        if cid not in cache_7b or len(cache_7b[cid]) != len(hyps):
            n_skip += 1; continue
        h7, r7 = pick_with_alpha_beta(hyps, cache_7b[cid],
                                       args.qwen7b_alpha, args.qwen7b_beta)
        e7 = edit_distance(ref_words, h7["text"].split())

        # 14B picked
        if cid not in cache_14b or len(cache_14b[cid]) != len(hyps):
            n_skip += 1; continue
        h14, r14 = pick_with_alpha_beta(hyps, cache_14b[cid],
                                         args.qwen14b_alpha, args.qwen14b_beta)
        e14 = edit_distance(ref_words, h14["text"].split())

        # Outcomes
        def outcome(rescored_errors, rescored_text, baseline_errors, baseline_text):
            if rescored_text == baseline_text: return "same"
            if rescored_errors < baseline_errors: return "rescued"
            if rescored_errors > baseline_errors: return "hurt"
            return "neutral"  # different text, same edit distance

        out_7b = outcome(e7, h7["text"], baseline_errors, baseline_text)
        out_14b = outcome(e14, h14["text"], baseline_errors, baseline_text)
        counts_7b[out_7b] += 1
        counts_14b[out_14b] += 1

        rows.append({
            "clip_id": cid,
            "ref": ref,
            "n_ref_words": len(ref_words),
            "baseline_text": baseline_text,
            "baseline_errors": baseline_errors,
            "qwen7b_text": h7["text"],
            "qwen7b_rank_picked": r7,
            "qwen7b_errors": e7,
            "qwen7b_outcome": out_7b,
            "qwen7b_delta": baseline_errors - e7,  # positive = rescued
            "qwen14b_text": h14["text"],
            "qwen14b_rank_picked": r14,
            "qwen14b_errors": e14,
            "qwen14b_outcome": out_14b,
            "qwen14b_delta": baseline_errors - e14,
        })

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

    # Summary
    n = len(rows)
    print("=" * 70)
    print(f"PER-CLIP ANALYSIS over {n} clips ({n_skip} skipped)")
    print("=" * 70)
    print()
    print(f"{'outcome':>10}  {'Qwen-7B':>10}  {'Qwen-14B':>10}")
    print(f"{'-'*10}  {'-'*10}  {'-'*10}")
    for k in ("rescued", "hurt", "same", "neutral"):
        pct7 = counts_7b[k] / n * 100
        pct14 = counts_14b[k] / n * 100
        print(f"{k:>10}  {counts_7b[k]:>5}({pct7:4.1f}%)  {counts_14b[k]:>5}({pct14:4.1f}%)")
    print()

    # Compute net WER delta
    net_7b = sum(r["qwen7b_delta"] for r in rows)
    net_14b = sum(r["qwen14b_delta"] for r in rows)
    total_words = sum(r["n_ref_words"] for r in rows)
    print(f"Net errors recovered by 7B:  {net_7b} (= {net_7b/total_words*100:.2f} WER points)")
    print(f"Net errors recovered by 14B: {net_14b} (= {net_14b/total_words*100:.2f} WER points)")
    print()

    # Top rescues by 7B
    print("=" * 70)
    print("TOP 10 RESCUES BY 7B (clips where 7B fixed the most errors)")
    print("=" * 70)
    top_resc_7b = sorted(rows, key=lambda r: -r["qwen7b_delta"])[:10]
    for r in top_resc_7b:
        if r["qwen7b_delta"] <= 0: break
        print(f"\n  clip: {r['clip_id']}  (Δ={r['qwen7b_delta']} errors fixed)")
        print(f"  REF:      {r['ref']}")
        print(f"  baseline: {r['baseline_text']}  (errors: {r['baseline_errors']})")
        print(f"  7B pick:  {r['qwen7b_text']}  (errors: {r['qwen7b_errors']}, rank {r['qwen7b_rank_picked']})")

    # Top hurts by 7B
    print()
    print("=" * 70)
    print("TOP 10 HURTS BY 7B (clips where 7B made things worse)")
    print("=" * 70)
    top_hurt_7b = sorted(rows, key=lambda r: r["qwen7b_delta"])[:10]
    for r in top_hurt_7b:
        if r["qwen7b_delta"] >= 0: break
        print(f"\n  clip: {r['clip_id']}  (Δ={r['qwen7b_delta']} more errors)")
        print(f"  REF:      {r['ref']}")
        print(f"  baseline: {r['baseline_text']}  (errors: {r['baseline_errors']})")
        print(f"  7B pick:  {r['qwen7b_text']}  (errors: {r['qwen7b_errors']}, rank {r['qwen7b_rank_picked']})")

    # Cases where 7B and 14B disagree
    print()
    print("=" * 70)
    print("TOP 10 CLIPS WHERE 7B AND 14B PICKED DIFFERENT (and disagree on quality)")
    print("=" * 70)
    disagree = [r for r in rows if r["qwen7b_text"] != r["qwen14b_text"]
                and r["qwen7b_errors"] != r["qwen14b_errors"]]
    disagree.sort(key=lambda r: -abs(r["qwen7b_errors"] - r["qwen14b_errors"]))
    for r in disagree[:10]:
        print(f"\n  clip: {r['clip_id']}  (7B errs={r['qwen7b_errors']}, 14B errs={r['qwen14b_errors']})")
        print(f"  REF:      {r['ref']}")
        print(f"  baseline: {r['baseline_text']}")
        print(f"  7B pick:  {r['qwen7b_text']}  (rank {r['qwen7b_rank_picked']})")
        print(f"  14B pick: {r['qwen14b_text']}  (rank {r['qwen14b_rank_picked']})")

    print()
    print("=" * 70)
    print(f"CSV saved: {args.out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
