import argparse, json, os, sys


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
    ap.add_argument("--nbest-jsonl", required=True)
    ap.add_argument("--cache-jsonl", required=True)
    ap.add_argument("--score-type", choices=["sum", "mean"], default="sum",
                    help="sum-logprob or mean-logprob (per-token).")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--betas", nargs="+", type=float,
                    default=[-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    with open(args.nbest_jsonl) as f:
        clips = [json.loads(l) for l in f]
    cache = {}
    with open(args.cache_jsonl) as f:
        for l in f:
            r = json.loads(l)
            cache[r["clip_id"]] = r["llm_scores"]

    print(f"Loaded {len(clips)} clips, {len(cache)} cached")
    print(f"Score type: {args.score_type}")
    print()

    print(f"{'alpha':>7} {'beta':>7} {'WER%':>8}")
    print("-" * 30)

    best = None
    for alpha in args.alphas:
        for beta in args.betas:
            errs, words = 0, 0
            for clip in clips:
                cid = clip["clip_id"]
                if cid not in cache: continue
                hyps = clip["hyps"]
                llm_raw = cache[cid]
                if len(llm_raw) != len(hyps): continue
                ref_words = clip["ref"].split()

                # Compute LM score (sum or mean) per hyp
                best_score = -float("inf")
                best_text = ""
                for h, raw_lm in zip(hyps, llm_raw):
                    n_tok = max(h["n_tokens"], 1)
                    if args.score_type == "mean":
                        lm = raw_lm / n_tok
                    else:
                        lm = raw_lm
                    n_words = len(h["text"].split())
                    score = h["visual_score"] + alpha * lm + beta * n_words
                    if score > best_score:
                        best_score = score
                        best_text = h["text"]

                err = edit_distance(ref_words, best_text.split())
                errs += err
                words += len(ref_words)

            wer = errs / max(words, 1)
            if best is None or wer < best[2]:
                best = (alpha, beta, wer)
            # only print if WER < 38% to keep table readable, plus alpha=0
            if wer < 0.385 or (alpha == 0.0 and beta == 0.0):
                print(f"{alpha:>7.2f} {beta:>7.2f} {wer*100:>7.2f}%")

    print()
    print(f"BEST: alpha={best[0]}, beta={best[1]}, WER={best[2]*100:.2f}%")


if __name__ == "__main__":
    main()
