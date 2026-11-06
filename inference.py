import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.io
PROJECT_ROOT = os.environ.get("VSR_PROJECT_ROOT", ".")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
from datamodule.transforms import VideoTransform
from infer_module import InferenceModule
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

DEFAULT_VISUAL_CKPT = os.environ.get("VSR_VISUAL_CKPT", "models/best.pth")
DEFAULT_LM_PATH = os.environ.get("VSR_LM_PATH", "qwen3b_model")

class VSRPipeline:
    # Wraps the full pipeline. Construct once, reuse for every video.
    def __init__(
        self,
        visual_ckpt: str = None,
        lm_path: str = None,
        beam_size: int = 25,
        ctc_weight: float = 0.1,
        lm_alpha: float = 0.05,
        lm_beta: float = -0.5,
        device: str = None,
    ):
        self.visual_ckpt = visual_ckpt or DEFAULT_VISUAL_CKPT
        self.lm_path = lm_path or DEFAULT_LM_PATH
        self.beam_size = beam_size
        self.ctc_weight = ctc_weight
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._video_proc = None
        self._landmarks_detector = None
        self._visual_module = None
        self._video_transform = None
        self._lm = None
        self._lm_tokenizer = None

        # Validate that local files exist
        if not os.path.isfile(self.visual_ckpt):
            raise FileNotFoundError(
                f"Visual checkpoint not found: {self.visual_ckpt}\n"
                f"Set VSR_VISUAL_CKPT or pass visual_ckpt= explicitly."
            )
        if not os.path.isdir(self.lm_path):
            raise FileNotFoundError(
                f"Local LM folder not found: {self.lm_path}\n"
                f"Download Qwen2.5-3B to this folder, or set VSR_LM_PATH.\n"
                f"To download: huggingface-cli download Qwen/Qwen2.5-3B "
                f"--local-dir {self.lm_path} --local-dir-use-symlinks False"
            )
        # Sanity check the LM folder has expected files
        required_lm_files = ["config.json", "tokenizer_config.json"]
        for fn in required_lm_files:
            if not os.path.isfile(os.path.join(self.lm_path, fn)):
                raise FileNotFoundError(
                    f"LM folder is missing {fn}. Files present: "
                    f"{os.listdir(self.lm_path)}\n"
                    f"Make sure you downloaded the full Qwen2.5-3B model."
                )

    # Lazy loaders 
    def _ensure_video_processor(self):
        if self._video_proc is None:
            print("[VSR] loading MediaPipe face/mouth detector...", flush=True)
            self._landmarks_detector = LandmarksDetector()
            self._video_proc = VideoProcess(convert_gray=True)

    def _ensure_visual_model(self):
        if self._visual_module is None:
            print(f"[VSR] loading visual model: {self.visual_ckpt}", flush=True)
            self._visual_module = InferenceModule(
                modality="video",
                ckpt_path=self.visual_ckpt,
                beam_size=self.beam_size,
                ctc_weight=self.ctc_weight,
                #lm_ckpt_path=None,
                #lm_weight=0.0,
                penalty=0.0,
            )
            self._visual_module.to(self.device).eval()
            self._video_transform = VideoTransform(subset="test")

    def _ensure_lm(self):
        if self._lm is None:
            print(f"[VSR] loading LM from local path: {self.lm_path}", flush=True)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._lm_tokenizer = AutoTokenizer.from_pretrained(
                self.lm_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.lm_path,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                trust_remote_code=True,
                local_files_only=True,
            )
            self._lm.eval()
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated() / 1e9
                print(f"[VSR] LM loaded. VRAM allocated: {vram_gb:.2f}GB", flush=True)
            else:
                print(f"[VSR] LM loaded on CPU", flush=True)

    def warmup(self):
        # Eagerly load all models.
        self._ensure_video_processor()
        self._ensure_visual_model()
        self._ensure_lm()

    # Stage 1: preprocess raw video, aligned 96x96 grayscale 
    def preprocess(self, video_path: str):
        # Returns (T, 1, 96, 96) tensor on CPU, or raises if face not found.
        self._ensure_video_processor()
        video_np, _, info = torchvision.io.read_video(video_path, pts_unit="sec")
        video_np = video_np.numpy()  # (T, H, W, C)
        if video_np.shape[0] == 0:
            raise RuntimeError("Video has no frames")

        # Step 1: detect facial landmarks per frame
        print(f"[VSR] detecting landmarks on {video_np.shape[0]} frames...", flush=True)
        landmarks = self._landmarks_detector(video_np)
        if landmarks is None:
            raise RuntimeError("Could not detect any faces in the video. "
                               "Make sure the speaker's face is clearly visible.")
        # Some frames may have None landmarks (face not detected). Check overall.
        n_detected = sum(1 for lm in landmarks if lm is not None)
        if n_detected == 0:
            raise RuntimeError("Face detection failed on every frame.")
        if n_detected < len(landmarks) * 0.5:
            print(f"[VSR] warning: face detected on only {n_detected}/{len(landmarks)} frames",
                  flush=True)

        # Step 2: align and crop mouth region using landmarks
        processed = self._video_proc(video_np, landmarks)
        if processed is None:
            raise RuntimeError("Mouth crop extraction failed (face landmarks too "
                               "sparse or invalid).")

        # processed shape: (T, H, W) grayscale, OR (T, H, W, C) — handle both
        if processed.ndim == 3:
            processed = processed[:, np.newaxis, :, :]
        elif processed.ndim == 4 and processed.shape[-1] == 1:
            # (T, H, W, 1) → (T, 1, H, W)
            processed = np.transpose(processed, (0, 3, 1, 2))
        return torch.from_numpy(processed).float()

    # Stage 2: visual model + beam search → n-best hypotheses 
    def visual_inference(self, video_tensor):
        """Returns list of dicts: [{rank, text, visual_score, n_tokens}]."""
        self._ensure_visual_model()
        x = self._video_transform(video_tensor).to(self.device)

        with torch.no_grad():
            x = self._visual_module.model.frontend(x.unsqueeze(0))
            x = self._visual_module.model.proj_encoder(x)
            enc, _ = self._visual_module.model.encoder(x, None)
            enc = enc.squeeze(0)
            nbest = self._visual_module.beam_search(enc)

        eos_id = self._visual_module.model.odim - 1
        text_transform = self._visual_module.text_transform
        hyps = []
        for rank, hyp in enumerate(nbest):
            d = hyp.asdict()
            tok_ids = [int(t) for t in d["yseq"][1:]]
            if tok_ids and tok_ids[-1] == eos_id:
                tok_ids = tok_ids[:-1]
            tok_tensor = torch.tensor(tok_ids, dtype=torch.long)
            text = text_transform.post_process(tok_tensor).replace("<eos>", "")
            text = " ".join(text.upper().split())
            hyps.append({
                "rank": rank + 1,
                "text": text,
                "visual_score": float(d["score"]),
                "n_tokens": len(tok_ids),
            })
        return hyps

    # Stage 3: LM rescore
    @torch.no_grad()
    def lm_rescore(self, hyps):
        # Returns (best_hyp_dict, llm_scores_list).
        self._ensure_lm()
        if not hyps:
            return None, []

        texts = [h["text"] for h in hyps]
        bos = self._lm_tokenizer.bos_token_id
        if bos is None:
            bos = self._lm_tokenizer.eos_token_id
        pad_id = self._lm_tokenizer.pad_token_id or self._lm_tokenizer.eos_token_id

        enc_list = []
        for txt in texts:
            if not txt.strip():
                enc_list.append([bos]); continue
            ids = [bos] + self._lm_tokenizer.encode(txt, add_special_tokens=False)
            enc_list.append(ids[:128])

        batch_size = 8
        llm_scores = []
        for i in range(0, len(enc_list), batch_size):
            chunk = enc_list[i:i+batch_size]
            lengths = [len(e) for e in chunk]
            max_l = max(lengths)
            batch = torch.full((len(chunk), max_l), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((len(chunk), max_l), dtype=torch.long, device=self.device)
            for j, e in enumerate(chunk):
                batch[j, :len(e)] = torch.tensor(e, dtype=torch.long, device=self.device)
                attn[j, :len(e)] = 1
            out = self._lm(input_ids=batch, attention_mask=attn)
            logp = F.log_softmax(out.logits.float(), dim=-1)
            for j, length in enumerate(lengths):
                if length <= 1:
                    llm_scores.append(float("-inf")); continue
                pred_pos = torch.arange(0, length-1, device=self.device)
                tgt = batch[j, 1:length]
                token_logps = logp[j, pred_pos, tgt]
                llm_scores.append(token_logps.sum().item())

        best_score = -float("inf")
        best_hyp = None
        for h, lm_s in zip(hyps, llm_scores):
            n_words = len(h["text"].split())
            combined = h["visual_score"] + self.lm_alpha * lm_s + self.lm_beta * n_words
            if combined > best_score:
                best_score = combined
                best_hyp = h
        return best_hyp, llm_scores

    # Public: full pipeline 
    def transcribe(self, video_path: str) -> dict:
        timings = {}

        t0 = time.time()
        video_tensor = self.preprocess(video_path)
        timings["preprocess"] = time.time() - t0

        t0 = time.time()
        hyps = self.visual_inference(video_tensor)
        timings["visual"] = time.time() - t0

        baseline_text = hyps[0]["text"] if hyps else ""

        t0 = time.time()
        best_hyp, llm_scores = self.lm_rescore(hyps)
        timings["rescore"] = time.time() - t0

        final_text = best_hyp["text"] if best_hyp else baseline_text
        rescued = (final_text != baseline_text)
        rescued_rank = best_hyp["rank"] if best_hyp else 1

        return {
            "text": final_text,
            "baseline_text": baseline_text,
            "rescued_by_lm": rescued,
            "rescued_rank": rescued_rank,
            "n_hypotheses": len(hyps),
            "n_frames": video_tensor.shape[0],
            "timings": timings,
            "total_time": sum(timings.values()),
        }


# Module-level singleton for convenience 
_global_pipeline = None

def get_pipeline(**kwargs):
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = VSRPipeline(**kwargs)
    return _global_pipeline


def transcribe_video(video_path: str) -> dict:
    return get_pipeline().transcribe(video_path)


# CLI for testing 
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--visual-ckpt", default=DEFAULT_VISUAL_CKPT)
    ap.add_argument("--lm-path", default=DEFAULT_LM_PATH)
    ap.add_argument("--lm-alpha", type=float, default=0.10)
    ap.add_argument("--lm-beta", type=float, default=0.0)
    args = ap.parse_args()

    pipeline = VSRPipeline(
        visual_ckpt=args.visual_ckpt,
        lm_path=args.lm_path,
        lm_alpha=args.lm_alpha,
        lm_beta=args.lm_beta,
    )

    print(f"\n=== Transcribing {args.video} ===")
    result = pipeline.transcribe(args.video)
    print(f"\nFinal:    {result['text']}")
    print(f"Baseline: {result['baseline_text']}")
    print(f"Rescued by LM: {result['rescued_by_lm']} (picked rank {result['rescued_rank']})")
    print(f"Time: preprocess={result['timings']['preprocess']:.2f}s  "
          f"visual={result['timings']['visual']:.2f}s  "
          f"rescore={result['timings']['rescore']:.2f}s  "
          f"total={result['total_time']:.2f}s")
