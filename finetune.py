"""
finetune.py 
learning rate : 1e-5
with Curriculum Learning and Temporal Consistency Loss
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from datamodule.transforms import VideoTransform, TextTransform
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from cosine import WarmupCosineScheduler


# Dataset for loading preprocessed LRS2 video clips and transcripts

class LRS2PreprocessedDataset(Dataset):
    # Initialize dataset metadata and preprocessing transforms
    def __init__(self, labels_file, preprocessed_root, subset):
        super().__init__()
        assert subset in ("train", "val")
        self.preprocessed_root = preprocessed_root
        self.video_transform = VideoTransform(subset="train" if subset == "train" else "test")
        self.text_transform  = TextTransform()
        self.entries = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                clip_id, n_frames, transcript = parts
                self.entries.append((clip_id, int(n_frames), transcript.rstrip("\r")))
        print(f"[dataset] {subset}: {len(self.entries)} clips from {labels_file}")

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.entries)

    # Return the original number of frames for a sample
    def get_n_frames(self, idx):
        return self.entries[idx][1]

    # Load a video sample, apply transforms, and tokenize its transcript
    def __getitem__(self, idx):
        clip_id, _, transcript = self.entries[idx]
        mp4_path = os.path.join(self.preprocessed_root, f"{clip_id}.mp4")
        try:
            video_np = torchvision.io.read_video(mp4_path, pts_unit="sec")[0].numpy()
        except Exception:
            return self.__getitem__((idx + 1) % len(self.entries))
        if video_np.shape[0] == 0:
            return self.__getitem__((idx + 1) % len(self.entries))

        video_tensor = torch.tensor(video_np).permute(0, 3, 1, 2)
        video_tensor = self.video_transform(video_tensor)
        tokens = self.text_transform.tokenize(transcript)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        return {"video": video_tensor, "tokens": tokens,
                "n_frames": video_tensor.size(0), "clip_id": clip_id}

# Sampler that groups clips by similar lengths with optional curriculum learning
class CurriculumFrameBucketSampler(Sampler):
    # Initialize batching strategy and curriculum settings
    def __init__(self, dataset, max_frames_per_batch,
                 curriculum_epochs=2, shuffle=True, seed=0):
        self.dataset           = dataset
        self.max_frames        = max_frames_per_batch
        self.curriculum_epochs = curriculum_epochs
        self.shuffle           = shuffle
        self.seed              = seed
        self.epoch             = 0
        self.lengths           = [dataset.get_n_frames(i) for i in range(len(dataset))]

    # Update the current epoch for curriculum scheduling
    def set_epoch(self, epoch):
        self.epoch = epoch

    # Generate batches with dynamic frame-based sizing
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.dataset)))

        if self.epoch < self.curriculum_epochs:
            idx.sort(key=lambda i: self.lengths[i])
        elif self.shuffle:
            jitter = [rng.random() * 5 for _ in idx]
            idx.sort(key=lambda i: self.lengths[i] + jitter[i])
        else:
            idx.sort(key=lambda i: self.lengths[i])

        batches, cur, cur_max = [], [], 0
        for i in idx:
            new_max   = max(cur_max, self.lengths[i])
            new_total = new_max * (len(cur) + 1)
            if cur and new_total > self.max_frames:
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)

        if self.shuffle and self.epoch >= self.curriculum_epochs:
            rng.shuffle(batches)

        for b in batches:
            yield b

    # Estimate the number of batches per epoch
    def __len__(self):
        return max(1, sum(self.lengths) // self.max_frames)

# Pad variable-length videos and token sequences into a batch
def collate(batch):
    videos   = [b["video"]  for b in batch]
    tokens   = [b["tokens"] for b in batch]
    n_frames = torch.tensor([b["n_frames"] for b in batch], dtype=torch.long)
    T_max    = max(v.size(0) for v in videos)
    B        = len(videos)
    vpad     = torch.zeros(B, T_max, 1, 88, 88, dtype=videos[0].dtype)
    for i, v in enumerate(videos):
        vpad[i, :v.size(0)] = v
    L_max = max(t.size(0) for t in tokens)
    tpad  = torch.full((B, L_max), -1, dtype=torch.long)
    for i, t in enumerate(tokens):
        tpad[i, :t.size(0)] = t
    return {"videos": vpad, "lengths": n_frames, "labels": tpad}


# Wrapper around the ESPNet E2E model with temporal consistency regularization
class E2EWithTemporalConsistency(torch.nn.Module):
    # Initialize the base model and temporal consistency settings
    def __init__(self, odim, modality, ctc_weight, temporal_weight=0.01):
        super().__init__()
        self.e2e             = E2E(odim=odim, modality=modality, ctc_weight=ctc_weight)
        self.temporal_weight = temporal_weight

    # Compute the main ASR loss and temporal smoothness loss
    def forward(self, videos, lengths, labels):
        loss_e2e, loss_ctc, loss_att, acc = self.e2e(videos, lengths, labels)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x = self.e2e.frontend(videos)
            x = self.e2e.proj_encoder(x)
            enc_feat, _ = self.e2e.encoder(x, None)
            temporal_diff = (enc_feat[:, 1:, :] - enc_feat[:, :-1, :]).pow(2).mean()
            loss_temporal = self.temporal_weight * temporal_diff

        loss_total = loss_e2e + loss_temporal
        return loss_total, loss_ctc, loss_att, acc, loss_temporal
    
    # Load pretrained weights into the underlying E2E model
    def load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        missing, unexpected = self.e2e.load_state_dict(sd, strict=False)
        if not missing and not unexpected:
            print("[ok] Pretrained weights loaded cleanly.")
    # Return only the internal E2E model weights for checkpoint saving
    def state_dict_e2e(self):
        return self.e2e.state_dict()

# Build the model, load pretrained weights, and move it to the target device
def build_model(args, device):
    text_transform = TextTransform()
    odim  = len(text_transform.token_list)
    model = E2EWithTemporalConsistency(
        odim=odim, modality="video",
        ctc_weight=args.ctc_weight,
        temporal_weight=args.temporal_weight,
    )
    print(f"[model] Loading pretrained: {args.pretrained_checkpoint}")
    model.load_pretrained(args.pretrained_checkpoint)
    print(f"[model] temporal_weight={args.temporal_weight}")
    return model.to(device)


# Evaluate the model on the validation set
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total = defaultdict(float)
    n = 0
    for batch in val_loader:
        videos  = batch["videos"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels  = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_ctc, loss_att, acc, loss_temp = model(videos, lengths, labels)
        total["loss"]      += loss.item()
        total["loss_ctc"]  += loss_ctc.item()
        total["loss_att"]  += loss_att.item()
        total["acc"]       += acc
        total["loss_temp"] += loss_temp.item()
        n += 1
    n = max(n, 1)
    return {k: v / n for k, v in total.items()}



# train ->Run the full training pipeline
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[train] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = LRS2PreprocessedDataset(args.train_labels, args.preprocessed_root, "train")
    val_ds   = LRS2PreprocessedDataset(args.val_labels,   args.preprocessed_root, "val")

    train_sampler = CurriculumFrameBucketSampler(
        train_ds, max_frames_per_batch=args.max_frames_per_batch,
        curriculum_epochs=args.warmup_epochs, shuffle=True, seed=args.seed,
    )
    val_sampler = CurriculumFrameBucketSampler(
        val_ds, max_frames_per_batch=args.max_frames_per_batch,
        curriculum_epochs=0, shuffle=False,
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds, batch_sampler=val_sampler,
                              num_workers=max(1, args.num_workers // 2), collate_fn=collate,
                              pin_memory=True, persistent_workers=args.num_workers > 0)

    model = build_model(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.max_epochs, steps_per_epoch,
    )

    global_step   = 0
    best_val_loss = float("inf")
    log_f = open(os.path.join(args.output_dir, "train_log.jsonl"), "a", encoding="utf-8")

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running, running_n = defaultdict(float), 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.max_epochs}")
        for batch_idx, batch in enumerate(pbar):
            videos  = batch["videos"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels  = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, loss_ctc, loss_att, acc, loss_temp = model(videos, lengths, labels)
                loss_to_backward = loss / args.grad_accum

            loss_to_backward.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running["loss"]      += loss.item()
            running["loss_ctc"]  += loss_ctc.item()
            running["loss_att"]  += loss_att.item()
            running["acc"]       += acc
            running["loss_temp"] += loss_temp.item()
            running_n += 1

            if batch_idx % 20 == 0:
                pbar.set_postfix(
                    loss=f"{running['loss']/running_n:.3f}",
                    temp=f"{running['loss_temp']/running_n:.4f}",
                    acc=f"{running['acc']/running_n:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        train_metrics = {k: v / max(running_n, 1) for k, v in running.items()}
        elapsed = time.time() - t0
        print(f"[train] epoch {epoch+1} done in {elapsed/60:.1f}min  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"temp={train_metrics['loss_temp']:.5f}  "
              f"acc={train_metrics['acc']:.4f}")

        val_metrics = validate(model, val_loader, device)
        print(f"[val]   epoch {epoch+1}  loss={val_metrics['loss']:.4f}  "
              f"temp={val_metrics['loss_temp']:.5f}  acc={val_metrics['acc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict_e2e(), os.path.join(args.output_dir, "best.pth"))
            print(f"[ckpt] *** NEW BEST (val_loss={best_val_loss:.4f})")

        log_f.write(json.dumps({
            "epoch": epoch + 1, "global_step": global_step,
            "elapsed_min": elapsed / 60,
            "train": train_metrics, "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }) + "\n")
        log_f.flush()

    log_f.close()
    print(f"\n[train] DONE — best val_loss: {best_val_loss:.4f}")

# parse_args -> Parse command-line arguments and return training configuration
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-root",    required=True)
    ap.add_argument("--train-labels",          required=True)
    ap.add_argument("--val-labels",            required=True)
    ap.add_argument("--pretrained-checkpoint", required=True)
    ap.add_argument("--output-dir",            required=True)
    ap.add_argument("--lr",                type=float, default=1e-5)
    ap.add_argument("--weight-decay",      type=float, default=0.03)
    ap.add_argument("--warmup-epochs",     type=int,   default=2)
    ap.add_argument("--max-epochs",        type=int,   default=20)
    ap.add_argument("--ctc-weight",        type=float, default=0.1)
    ap.add_argument("--grad-clip",         type=float, default=1.0)
    ap.add_argument("--grad-accum",        type=int,   default=1)
    ap.add_argument("--max-frames-per-batch", type=int, default=2400)
    ap.add_argument("--num-workers",       type=int,   default=4)
    ap.add_argument("--seed",              type=int,   default=0)
    ap.add_argument("--temporal-weight",   type=float, default=0.01)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)


if __name__ == "__main__":
    main()
