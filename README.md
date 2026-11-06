# Lip Reading System

**Sentence-level visual speech recognition with LLM rescoring on LRS2.**

A deep learning system that reads spoken sentences from silent video — no audio required. Built as a graduation project at Princess Sumaya University for Technology.

The system combines a fine-tuned Conformer-based visual encoder with a Qwen2.5 large language model rescorer to transcribe what a person is saying from lip movements alone. On the official LRS2 test set, it achieves **26.81% Word Error Rate**, down from 39.02% with the untuned pretrained baseline (a **31% relative reduction**).

**Highlights**

- Preprocessing pipeline for **220+ hours** of LRS2 video: MediaPipe face detection, landmark smoothing, affine alignment and 96×96 mouth-region cropping
- Fine-tuning with **curriculum learning** (frame-length bucketing) and a **temporal-consistency loss**: 39.02% → 28.79% WER
- **N-best rescoring with Qwen2.5 LLMs** (3B / 7B, bf16 / int8 / int4) via single-pass teacher forcing: 28.79% → 26.81% WER
- End-to-end **Streamlit demo** that transcribes an uploaded video in about 12 seconds on an RTX 4070


## Overview

Lip reading — also called visual speech recognition (VSR) — is the task of transcribing speech from a silent video of a person speaking, without relying on the audio signal. Applications include silent communication contexts, noisy environments where audio is unreliable, and research foundations for future assistive technologies.

This project implements a sentence-level VSR system on the **LRS2** benchmark (Lip Reading Sentences 2), a dataset of BBC television broadcasts collected by the University of Oxford. Given a silent video clip as input, the system:

1. Detects and aligns the speaker's mouth region using MediaPipe
2. Encodes the lip motion using a Conformer-based deep neural network
3. Decodes candidate transcriptions via hybrid CTC/attention beam search
4. Re-ranks the candidates using **Qwen2.5** as a language-model rescorer
5. Returns the most plausible transcription

The full system runs end-to-end through a browser-based Streamlit interface or directly from the command line.

---

## How it works

The pipeline has four sequential stages.

### Stage 1 — Preprocessing

The input video is decoded into RGB frames. For each frame, MediaPipe's `FaceDetection` model locates four facial keypoints (right eye, left eye, nose tip, mouth center). Missing detections are linearly interpolated between adjacent frames. The keypoints are then smoothed across a 12-frame window and used to estimate a similarity affine transform that maps each frame onto a canonical 256×256 mean-face template. A 96×96 patch is cropped around the mouth keypoint and converted to grayscale, producing a tensor of shape `(T, 96, 96)`.

### Stage 2 — Visual encoding

The mouth tensor passes through a four-block visual network:

| Block | Purpose | Output shape |
|---|---|---|
| **3D-Convolutional frontend** (kernel 5×7×7) | Captures short-term lip motion | `(T, 64, …)` |
| **ResNet-18 spatial trunk** | Per-frame visual features | `(T, 512)` |
| **Linear projection** | Lifts to model dimension | `(T, 768)` |
| **Conformer encoder** (12 layers, 12 heads, depthwise conv kernel 31) | Contextual representations combining local + long-range temporal dependencies | `(T, 768)` |

### Stage 3 — Hybrid CTC/attention decoding

The encoder output feeds two heads in parallel:

- A **Transformer decoder** (6 layers, 12 heads, dim 768) for autoregressive token prediction
- A **CTC head** for monotonic alignment

Hybrid beam search combines both scores (CTC weight 0.1) and produces an **n-best list of 25 candidate transcriptions**, each annotated with its visual log-probability.

### Stage 4 — LLM rescoring

The 25 candidates are scored by **Qwen2.5** (3B for the deployed demo, 7B for the best-accuracy configuration). For each candidate `c`, the LLM computes the sum of token log-probabilities in a single teacher-forced forward pass:

```
log P_LM(c) = Σᵢ log P_LM(tᵢ | t<ᵢ)
```

No autoregressive generation is needed since the candidate is already known. Candidates are then re-ranked by a linear score combination:

```
score(c) = visual_score(c) + α · log P_LM(c) + β · |c|_words
```

with α = 0.10 and β = 0.0 (chosen by grid search on the validation set). The candidate with the highest combined score is returned as the final transcription.

---

## Results

All numbers are **Word Error Rate (WER)** on the official LRS2 test split (1,243 clips), measured with beam size 25 and CTC weight 0.1.

| Configuration | Test WER |
|---|---|
| Pretrained Auto-AVSR (zero-shot, no fine-tuning) | 39.02% |
| Standard fine-tuning on LRS2 | 32.33% |
| Fine-tuning + curriculum learning + temporal-consistency loss | 28.79% |
| + Qwen2.5-3B rescoring (α=0.10, β=0.0) | 28.29% |
| **+ Qwen2.5-7B rescoring (α=0.20, β=0.20)** | **26.81%** |

For reference, the oracle WER on the beam=25 n-best list is **11.54%** — the theoretical ceiling if rescoring could always pick the best candidate from the visual model's output. This means roughly 13% of the available rescoring headroom is recovered by Qwen-7B.

---

## Installation

### Prerequisites

- Python **3.10** or **3.11**
- NVIDIA GPU with at least **8 GB VRAM** (24 GB recommended for Qwen-7B/14B experiments)
- At least **10 GB** free disk space (15 GB if including the Qwen-3B local weights)
- Operating system: Windows 10/11 or Linux

### Setup

1. Clone or download this repository to your machine.

2. Create and activate a Python virtual environment:

   ```bash
   # Windows
   python -m venv vsr_env
   .\vsr_env\Scripts\activate

   # Linux / macOS
   python -m venv vsr_env
   source vsr_env/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the model files in the project root:

   - `models/best.pth` — the fine-tuned visual checkpoint
   - `models/vsr_trlrs3vox2_base.pth` — the original Auto-AVSR pretrained checkpoint (only needed if you plan to retrain)
   - `qwen3b_model/` — the local Qwen2.5-3B model directory (contains tokenizer + safetensors weights)

   The weights are not included in this repository because of their size. Contact the authors for access to the fine-tuned checkpoint. The Qwen2.5 weights are available from the Qwen organization on Hugging Face.

   Paths can be overridden with environment variables: `VSR_VISUAL_CKPT` for the visual checkpoint and `VSR_LM_PATH` for the LLM directory. Note that `app.py` loads `models/avg_e4_e9.pth` (checkpoint averaged over epochs 4–9) by default, while `inference.py` loads `models/best.pth`.

5. Verify the installation:

   ```bash
   python -c "from inference import VSRPipeline; p = VSRPipeline(); p.warmup(); print('Setup complete')"
   ```

   If this prints `Setup complete` with no errors, you're ready to go.

---

## Usage

### Streamlit web demo

Launch the browser interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload an MP4, AVI, or MOV file of a person speaking facing the camera, then click the inference button. Expect ~12 seconds per video on an RTX 4070.

**Tips for best recognition:**
- The speaker's face should be clearly visible throughout the clip
- The mouth should not be covered or occluded
- The speaker should face the camera as directly as possible
- Lighting should be stable and bright enough to show lip movement
- The clip should contain a single spoken sentence

### Command-line inference

To transcribe a single video without the UI:

```bash
python inference.py path/to/video.mp4
```

The transcript is printed to stdout.

### Programmatic use

```python
from inference import VSRPipeline

pipeline = VSRPipeline()
pipeline.warmup()  # Eagerly load all models into VRAM

result = pipeline.transcribe("path/to/video.mp4")

print(result["text"])              # Final transcription (after LLM rescoring)
print(result["visual_only"])       # Visual-only baseline (before rescoring)
print(result["rescued_by_lm"])     # True if LLM changed the prediction
print(result["n_frames"])          # Number of frames processed
print(result["elapsed_seconds"])   # Per-stage timing breakdown
```

---


## Training from scratch

If you have access to the LRS2 dataset and want to reproduce training:

### 1. Preprocess raw LRS2 videos into grayscale mouth crops

```bash
python preprocess_lrs2.py --input-root /path/to/LRS2/main --output-root /path/to/main_mouth96
```

This processes the train (F1–F34), val and test splits; use `--splits train val` to run only some of them. Already-processed clips are skipped, so the script can be resumed. Each split takes several hours on a single workstation.

### 2. Generate label files

```bash
python make_labels.py --preprocessed-root /path/to/main_mouth96 --out-dir labels
```

This writes `labels/train_labels.txt`, `labels/val_labels.txt` and `labels/test_labels.txt` (tab-separated `<clip_id>\t<num_frames>\t<transcript>`). Train/val clips are filtered to 5–600 frames; the test set is kept complete.

### 3. Fine-tune the visual model

```bash
python finetune.py \
    --preprocessed-root /path/to/main_mouth96 \
    --train-labels labels/train_labels.txt \
    --val-labels labels/val_labels.txt \
    --pretrained-checkpoint models/vsr_trlrs3vox2_base.pth \
    --output-dir runs/exp01 \
    --max-epochs 20 \
    --lr 1e-5 \
    --temporal-weight 0.01 \
    --warmup-epochs 2
```

Training takes approximately 24-48 hours on a single RTX 4090, depending on disk I/O.

### 4. Evaluate visual-only WER

```bash
python eval_visual_wer.py \
    --ckpt runs/exp01/best.pth \
    --labels labels/test_labels.txt \
    --root /path/to/main_mouth96/test \
    --out-csv results_test.csv
```

### 5. Generate n-best and run LLM rescoring

```bash
python generate_nbest_v2.py \
    --ckpt runs/exp01/best.pth \
    --labels-file labels/test_labels.txt \
    --preprocessed-root /path/to/main_mouth96/test \
    --beam-size 25 \
    --output-jsonl results/test_beam25.jsonl

python rescore_with_llm_quant.py \
    --nbest-jsonl results/test_beam25.jsonl \
    --model-name qwen3b_model \
    --quant-mode bf16 \
    --batch-size 16 \
    --cache-jsonl results/test_qwen3b_scores.jsonl \
    --out-prefix results/test_qwen3b
```

---
To try other α/β weights without re-running the LLM, re-sweep the cached scores:

```bash
python resweep_cached.py \
    --nbest-jsonl results/test_beam25.jsonl \
    --cache-jsonl results/test_qwen3b_scores.jsonl
```

`error_analysis.py` compares the visual-only, Qwen-7B and Qwen-14B outputs clip by clip.

---

## Repository structure

```
.
├── app.py                      # Streamlit web demo
├── inference.py                # End-to-end pipeline: video → mouth crop → n-best → LLM rescoring
├── infer_module.py             # Visual model + hybrid CTC/attention beam search wrapper
├── preprocess_lrs2.py          # Raw LRS2 video → 96×96 grayscale mouth crops
├── make_labels.py              # Builds train/val/test label files
├── finetune.py                 # Fine-tuning with curriculum sampling + temporal-consistency loss
├── cosine.py                   # Warmup + cosine learning-rate schedule
├── eval_lrs2_preprocessed.py   # Shared evaluation helpers
├── eval_visual_wer.py          # Visual-only WER evaluation
├── generate_nbest_v2.py        # Writes n-best hypotheses to JSONL
├── rescore_with_llm_quant.py   # Qwen2.5 rescoring (bf16 / int8 / int4)
├── resweep_cached.py           # Re-tunes α/β from cached LLM scores
├── error_analysis.py           # Per-clip comparison of rescoring configurations
├── transformer_lm_scorer.py    # Experimental: small Transformer LM for shallow fusion during beam search
├── espnet/                     # Conformer / Transformer / CTC / beam search (from Auto-AVSR / ESPnet)
├── datamodule/                 # Video and text transforms (from Auto-AVSR)
├── preparation/                # MediaPipe detector and mouth-crop alignment (from Auto-AVSR)
└── spm/                        # SentencePiece unigram vocabulary (5,000 units)
```

---

## Acknowledgements

- The visual model architecture, pretrained checkpoint, and the `espnet/`, `datamodule/` and `preparation/` modules come from **[Auto-AVSR](https://github.com/mpc001/auto_avsr)** (Ma et al., ICASSP 2023), which builds on **[ESPnet](https://github.com/espnet/espnet)**. These files keep their original licenses.
- **LRS2** was collected by the Visual Geometry Group, University of Oxford, from BBC broadcasts. The dataset is not redistributed here; request access from its maintainers.
- **Qwen2.5** language models by the Qwen team.

## Team

Graduation project, B.Sc. in Data Science and Artificial Intelligence, Princess Sumaya University for Technology (2026).

- **Abdullah Al-Saidi**: data preprocessing pipeline, model training experiments
- **Feras Nassar**: model architecture design, implementation, and training experiments
- **Musab Farhan**: interface design and development
