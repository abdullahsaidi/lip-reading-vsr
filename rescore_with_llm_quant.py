#This file test how quantization affects rescoring quality
import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


@torch.no_grad()
def score_batch(tokenizer, model, texts, device, max_len=128):
    out_scores = []
    if not texts:
        return out_scores
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    enc_list = []
    for txt in texts:
        if not txt.strip():
            enc_list.append([bos])
            continue
        ids = tokenizer.encode(txt, add_special_tokens=False)
        ids = [bos] + ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        enc_list.append(ids)
    lengths = [len(e) for e in enc_list]
    max_l = max(lengths)
    pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    batch = torch.full((len(enc_list), max_l), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(enc_list), max_l), dtype=torch.long, device=device)
    for i, e in enumerate(enc_list):
        batch[i, :len(e)] = torch.tensor(e, dtype=torch.long, device=device)
        attention_mask[i, :len(e)] = 1
    outputs = model(input_ids=batch, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    for i, length in enumerate(lengths):
        if length <= 1:
            out_scores.append(float("-inf"))
            continue
        pred_positions = torch.arange(0, length - 1, device=device)
        target_tokens = batch[i, 1:length]
        token_logps = log_probs[i, pred_positions, target_tokens]
        out_scores.append(token_logps.sum().item())
    return out_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest-jsonl", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--quant-mode", choices=["bf16", "int8", "int4"], default="bf16",
                    help="Quantization mode for the LM.")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=-1)
    ap.add_argument("--cache-jsonl", default=None)
    ap.add_argument("--out-prefix", default="results/qwen_quant")
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[rescore] device: {device}, quant_mode: {args.quant_mode}", flush=True)

    with open(args.nbest_jsonl) as f:
        clips = [json.loads(line) for line in f]
    if args.max_clips > 0:
        clips = clips[:args.max_clips]
    n_total_hyps = sum(len(c["hyps"]) for c in clips)
    print(f"[rescore] {len(clips)} clips, {n_total_hyps} total hypotheses", flush=True)

    cached_scores = None
    if args.cache_jsonl and os.path.exists(args.cache_jsonl):
        print(f"[rescore] loading cached LLM scores from {args.cache_jsonl}")
        cached_scores = {}
        with open(args.cache_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                cached_scores[rec["clip_id"]] = rec["llm_scores"]
        missing = [c["clip_id"] for c in clips if c["clip_id"] not in cached_scores]
        if missing:
            print(f"[rescore] cache missing {len(missing)} clips, will recompute all")
            cached_scores = None

    if cached_scores is None:
        print(f"[rescore] loading {args.model_name} in {args.quant_mode} ...", flush=True)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

        # Quantization setup
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": {"": device},
        }
        if args.quant_mode == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif args.quant_mode == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif args.quant_mode == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        model.eval()

        # Print actual VRAM usage so we know it'd fit on 4070 8GB
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[rescore] loaded in {time.time()-t0:.1f}s. params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        print(f"[rescore] VRAM allocated: {vram_used:.2f}GB, reserved: {vram_reserved:.2f}GB")

        cached_scores = {}
        cache_f = None
        if args.cache_jsonl:
            os.makedirs(os.path.dirname(args.cache_jsonl), exist_ok=True)
            cache_f = open(args.cache_jsonl, "w", encoding="utf-8")

        t0 = time.time()
        n_scored = 0
        for ci, clip in enumerate(clips):
            texts = [h["text"] for h in clip["hyps"]]
            llm_scores = []
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i+args.batch_size]
                scores = score_batch(tokenizer, model, batch, device)
                llm_scores.extend(scores)
            cached_scores[clip["clip_id"]] = llm_scores
            n_scored += len(texts)
            if cache_f:
                cache_f.write(json.dumps({
                    "clip_id": clip["clip_id"],
                    "llm_scores": llm_scores,
                }) + "\n")
                cache_f.flush()
            if (ci+1) % 10 == 0 or ci+1 == len(clips):
                elapsed = time.time() - t0
                rate = n_scored / max(elapsed, 1e-9)
                eta = (n_total_hyps - n_scored) / max(rate, 1e-9) / 60
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"[rescore] {ci+1}/{len(clips)} clips, {n_scored}/{n_total_hyps} hyps  "
                      f"rate={rate:.0f} hyp/s  eta={eta:.1f}min  vram={vram:.1f}GB", flush=True)
        if cache_f:
            cache_f.close()
        del model
        torch.cuda.empty_cache()
        print(f"[rescore] LLM scoring done in {(time.time()-t0)/60:.1f}min", flush=True)

    # Sweep
    print(f"\n[rescore] sweeping alpha={args.alphas}, beta={args.betas}", flush=True)
    print(f"{'alpha':>7} {'beta':>7} {'WER%':>8} {'errors':>10} {'words':>10}")
    print("-" * 50)

    results = []
    best = None
    for alpha in args.alphas:
        for beta in args.betas:
            total_errors = 0
            total_words = 0
            for clip in clips:
                cid = clip["clip_id"]
                ref = clip["ref"]
                hyps = clip["hyps"]
                llm_scores = cached_scores[cid]
                if len(llm_scores) != len(hyps):
                    continue
                best_score = -float("inf")
                best_hyp = None
                for h, llm in zip(hyps, llm_scores):
                    n_words = len(h["text"].split())
                    score = h["visual_score"] + alpha * llm + beta * n_words
                    if score > best_score:
                        best_score = score
                        best_hyp = h
                if best_hyp is None: continue
                err = edit_distance(ref.split(), best_hyp["text"].split())
                total_errors += err
                total_words += len(ref.split())
            wer = total_errors / max(total_words, 1)
            print(f"{alpha:>7.2f} {beta:>7.2f} {wer*100:>7.2f}% {total_errors:>10} {total_words:>10}")
            results.append({"alpha": alpha, "beta": beta, "wer": wer,
                            "errors": total_errors, "words": total_words})
            if best is None or wer < best["wer"]:
                best = results[-1]

    print()
    print("=" * 60)
    print(f"BEST: alpha={best['alpha']}, beta={best['beta']}, WER={best['wer']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
