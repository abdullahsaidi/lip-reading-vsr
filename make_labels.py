"""
make_labels.py

Builds label files for finetune.py from preprocessed LRS2 clips.
Scans main_mouth96/ directly 

Output:
    labels/train_labels.txt
    labels/val_labels.txt
    labels/test_labels.txt   (clip ids relative to <root>/test, no length filter)

Each line (TAB-separated, matches finetune.py parser exactly):
    <clip_id>\t<num_frames>\t<transcript>

Example:
    F1/5535415699068794046/00001\t75\tHE SAID THE MEETING WAS CANCELLED

Usage:
    python make_labels.py --preprocessed-root /path/to/main_mouth96 --out-dir labels
"""

import argparse
import os
import re
import cv2
from tqdm import tqdm

# Paths are set from the command line in main()
PREPROCESSED_ROOT = None
LABELS_OUT_DIR    = None

TRAIN_FOLDERS = [f"F{i}" for i in range(1, 35)]
VAL_FOLDER    = "val"

MIN_FRAMES = 5
MAX_FRAMES = 600

_CHARS_TO_IGNORE = r"[\,\?\.\!\-\;\:\"]"


def normalize_text(text):
    text = re.sub(_CHARS_TO_IGNORE, "", text)
    text = text.upper().replace("\u2019", "'")
    return " ".join(text.split())


def read_transcript(txt_path):
    try:
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return None
    for line in content.splitlines():
        if line.startswith("Text:"):
            return line[len("Text:"):].strip()
    return content.strip() or None


def count_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

# Returns list of (clip_id, mp4_path, txt_path).
def collect_clips(root, folder_names):
    clips = []
    for folder_name in folder_names:
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for speaker in sorted(os.listdir(folder_path)):
            speaker_path = os.path.join(folder_path, speaker)
            if not os.path.isdir(speaker_path):
                continue
            for fname in sorted(os.listdir(speaker_path)):
                if not fname.endswith(".mp4"):
                    continue
                clip_name = fname[:-4]
                mp4_path  = os.path.join(speaker_path, fname)
                txt_path  = os.path.join(speaker_path, clip_name + ".txt")
                if not os.path.isfile(txt_path):
                    continue
                clip_id = f"{folder_name}/{speaker}/{clip_name}"
                clips.append((clip_id, mp4_path, txt_path))
    return clips


def build_labels(clips, out_file, split_name, length_filter=True):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    stats = {"ok": 0, "missing": 0, "too_short": 0,
             "too_long": 0, "bad_transcript": 0}

    with open(out_file, "w", encoding="utf-8", newline="\n") as fout:
        pbar = tqdm(clips, desc=f"[{split_name}]", unit="clip", dynamic_ncols=True)
        for clip_id, mp4_path, txt_path in pbar:

            if not os.path.isfile(mp4_path) or not os.path.isfile(txt_path):
                stats["missing"] += 1
                continue

            n_frames = count_frames(mp4_path)
            if n_frames < (MIN_FRAMES if length_filter else 1):
                stats["too_short"] += 1
                continue
            if length_filter and n_frames > MAX_FRAMES:
                stats["too_long"] += 1
                continue

            transcript = read_transcript(txt_path)
            if not transcript:
                stats["bad_transcript"] += 1
                continue
            transcript = normalize_text(transcript)
            if not transcript:
                stats["bad_transcript"] += 1
                continue

            fout.write(f"{clip_id}\t{n_frames}\t{transcript}\n")
            stats["ok"] += 1

            skipped = sum(stats.values()) - stats["ok"]
            pbar.set_postfix({"written": stats["ok"], "skipped": skipped})

    print(f"\n  [{split_name}] Summary:")
    print(f"    Written:           {stats['ok']:>6}")
    print(f"    Missing files:     {stats['missing']:>6}")
    print(f"    Too short (<{MIN_FRAMES}f):  {stats['too_short']:>6}")
    print(f"    Too long  (>{MAX_FRAMES}f): {stats['too_long']:>6}")
    print(f"    Bad transcript:    {stats['bad_transcript']:>6}")
    print(f"    Output: {out_file}")
    return stats


def main():
    global PREPROCESSED_ROOT, LABELS_OUT_DIR
    ap = argparse.ArgumentParser(description="Build TSV label files from preprocessed LRS2 clips.")
    ap.add_argument("--preprocessed-root", required=True, help="Output folder of preprocess_lrs2.py")
    ap.add_argument("--out-dir", default="labels", help="Where to write *_labels.txt")
    args = ap.parse_args()
    PREPROCESSED_ROOT = args.preprocessed_root
    LABELS_OUT_DIR = args.out_dir

    print("=" * 60)
    print("  make_labels.py — Build Label Files for finetune.py")
    print("=" * 60)
    print(f"  Source : {PREPROCESSED_ROOT}")
    print(f"  Output : {LABELS_OUT_DIR}")

    if not os.path.isdir(PREPROCESSED_ROOT):
        print(f"\n[ERROR] Not found: {PREPROCESSED_ROOT}")
        return

    print(f"\n{'='*60}")
    print("  Collecting training clips (F1-F34) ...")
    train_clips = collect_clips(PREPROCESSED_ROOT, TRAIN_FOLDERS)
    print(f"  Found {len(train_clips)} clips")
    build_labels(
        train_clips,
        out_file=os.path.join(LABELS_OUT_DIR, "train_labels.txt"),
        split_name="TRAIN",
    )

    print(f"\n{'='*60}")
    print("  Collecting validation clips ...")
    val_clips = collect_clips(PREPROCESSED_ROOT, [VAL_FOLDER])
    print(f"  Found {len(val_clips)} clips")
    build_labels(
        val_clips,
        out_file=os.path.join(LABELS_OUT_DIR, "val_labels.txt"),
        split_name="VAL",
    )

    # Test labels: ids are relative to <root>/test (what eval_visual_wer.py and
    # generate_nbest_v2.py expect), and no length filter so the full test set is kept.
    print(f"\n{'='*60}")
    print("  Collecting test clips ...")
    test_clips = [
        (clip_id[len("test/"):], mp4, txt)
        for clip_id, mp4, txt in collect_clips(PREPROCESSED_ROOT, ["test"])
    ]
    print(f"  Found {len(test_clips)} clips")
    build_labels(
        test_clips,
        out_file=os.path.join(LABELS_OUT_DIR, "test_labels.txt"),
        split_name="TEST",
        length_filter=False,
    )

    print(f"\n{'='*60}")
    print("  Done. Next step: run finetune.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
