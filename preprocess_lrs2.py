"""
preprocess_lrs2.py
==================
Preprocesses raw LRS2 video clips to be compatible with finetune.py.

Expected input structure:
    main/
    ├── F1/ F2/ ... F34/    <- training folders
    ├── val/                <- validation folder
    └── test/               <- test folder

    Each folder contains:
        <speaker_id>/
            <clip_id>.mp4
            <clip_id>.txt

Output structure (exactly what finetune.py expects):
    main_mouth96/
    ├── F1/<speaker>/<clip>.mp4    <- 96x96 grayscale
    ├── F1/<speaker>/<clip>.txt    <- original transcript
    ├── ...
    ├── val/<speaker>/<clip>.mp4
    ├── val/<speaker>/<clip>.txt
    ├── test/<speaker>/<clip>.mp4
    └── test/<speaker>/<clip>.txt

Usage:
    python preprocess_lrs2.py --input-root /path/to/LRS2/main \
                              --output-root /path/to/main_mouth96 \
                              [--splits train val test]

Requirements:
    pip install mediapipe opencv-python scikit-image numpy tqdm
"""

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


# Paths are set from the command line in main()

INPUT_ROOT  = None
OUTPUT_ROOT = None

FPS = 25

TRAIN_FOLDERS = [f"F{i}" for i in range(1, 35)]
VAL_FOLDER    = "val"
TEST_FOLDER   = "test"



# Import original model files

_project_root = os.path.dirname(os.path.abspath(__file__))
_mediapipe_dir = os.path.join(_project_root, "preparation", "detectors", "mediapipe")
for _p in [_project_root, _mediapipe_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from preparation.detectors.mediapipe.detector import LandmarksDetector
    from preparation.detectors.mediapipe.video_process import VideoProcess
except ImportError:
    try:
        from detector import LandmarksDetector
        from video_process import VideoProcess
    except ImportError:
        print("[ERROR] Could not import detector.py or video_process.py.")
        print("        Make sure preprocess_lrs2.py is in the same folder as those files,")
        print("        or that the preparation/detectors/mediapipe/ path exists.")
        sys.exit(1)


# Read all frames from a video file and convert them from BGR to RGB
def read_video_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# Save a grayscale mouth sequence as a standard MP4 video
def save_grayscale_video(sequence, out_path, fps=25):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T, H, W = sequence.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for frame in sequence:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()

# Copy the transcript file to the processed output directory
def copy_transcript(src_txt, dst_txt):
    """Copy a transcript file as-is to the output location."""
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
    with open(src_txt, encoding="utf-8", errors="replace") as f:
        content = f.read()
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write(content)

# Collect all valid video-transcript pairs from a dataset split
def collect_clips(folder_path, split_prefix):
    clips = []
    if not os.path.isdir(folder_path):
        return clips

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
            clip_id = f"{split_prefix}/{speaker}/{clip_name}"
            clips.append((clip_id, mp4_path, txt_path))
    return clips

# Process a single clip: detect landmarks, crop the mouth region, and save outputs
def process_one_clip(mp4_path, txt_path, out_mp4, out_txt,
                     detector, video_processor):

    # Skip already-processed clips - supports resuming if interrupted
    if os.path.isfile(out_mp4) and os.path.isfile(out_txt):
        return True, "already_done"

    # Step 1: Read video frames
    frames = read_video_frames(mp4_path)
    if len(frames) == 0:
        return False, "empty_video"

    # Step 2: Detect face landmarks (4 keypoints: right eye, left eye, nose, mouth)
    try:
        landmarks = detector(frames)
    except AssertionError:
        return False, "no_face_detected"
    except Exception:
        return False, "detector_error"

    # Step 3: Affine transform + crop mouth region to 96x96 grayscale
    try:
        sequence = video_processor(frames, landmarks)
    except Exception:
        return False, "crop_failed"

    if sequence is None or len(sequence) == 0:
        return False, "crop_failed"

    # sequence shape: (T, 96, 96) — matches what finetune.py expects

    # Step 4: Save processed video
    save_grayscale_video(sequence, out_mp4, fps=FPS)

    # Step 5: Copy transcript
    copy_transcript(txt_path, out_txt)

    return True, "ok"

# Process all clips in a dataset split and track preprocessing statistics
def process_split(split_clips, split_label, detector, video_processor):
    """Process all clips in a split with a progress bar and statistics."""
    stats = {
        "ok": 0,
        "already_done": 0,
        "empty_video": 0,
        "no_face_detected": 0,
        "crop_failed": 0,
        "detector_error": 0,
    }

    print(f"\n{'='*60}")
    print(f"  Processing: {split_label}  ({len(split_clips)} clips)")
    print(f"{'='*60}")

    pbar = tqdm(split_clips, desc=split_label, unit="clip", dynamic_ncols=True)

    for clip_id, mp4_path, txt_path in pbar:
        out_mp4 = os.path.join(OUTPUT_ROOT, f"{clip_id}.mp4")
        out_txt = os.path.join(OUTPUT_ROOT, f"{clip_id}.txt")

        ok, reason = process_one_clip(
            mp4_path, txt_path, out_mp4, out_txt,
            detector, video_processor
        )

        stats[reason] = stats.get(reason, 0) + 1

        pbar.set_postfix({
            "done":    stats["ok"] + stats["already_done"],
            "no_face": stats["no_face_detected"],
            "failed":  stats["crop_failed"] + stats["empty_video"],
        })

    success = stats["ok"] + stats["already_done"]
    total   = len(split_clips)
    print(f"\n  [{split_label}] Summary:")
    print(f"    Processed (new):      {stats['ok']:>6}")
    print(f"    Skipped (done):       {stats['already_done']:>6}")
    print(f"    Failed - no face:     {stats['no_face_detected']:>6}")
    print(f"    Failed - empty video: {stats['empty_video']:>6}")
    print(f"    Failed - crop error:  {stats['crop_failed']:>6}")
    print(f"    Total success:        {success:>6} / {total}")
    return stats


# Main

def parse_args():
    ap = argparse.ArgumentParser(description="Crop 96x96 grayscale mouth ROIs from raw LRS2 videos.")
    ap.add_argument("--input-root", required=True, help="LRS2 'main' folder (contains F1..F34, val, test)")
    ap.add_argument("--output-root", required=True, help="Where to write the preprocessed clips")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"], help="Which splits to process")
    return ap.parse_args()


def main():
    global INPUT_ROOT, OUTPUT_ROOT
    args = parse_args()
    INPUT_ROOT = args.input_root
    OUTPUT_ROOT = args.output_root

    print("=" * 60)
    print("  LRS2 Preprocessing — compatible with finetune.py")
    print("=" * 60)
    print(f"  Input  : {INPUT_ROOT}")
    print(f"  Output : {OUTPUT_ROOT}")

    # Create output directories if they do not already exist
    if not os.path.isdir(INPUT_ROOT):
        print(f"\n[ERROR] Input directory not found:\n  {INPUT_ROOT}")
        sys.exit(1)

  
    print("\n[init] Loading LandmarksDetector and VideoProcess ...")
    # Initialize MediaPipe face landmark detection and mouth region processing
    detector = LandmarksDetector()
    video_processor = VideoProcess(
        mean_face_path=os.path.join(_mediapipe_dir, "20words_mean_face.npy"),
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
        convert_gray=True,
    )
    print("[init] Ready\n")

    overall_stats = {}

    # Training: F1 to F34
    if "train" in args.splits:
        train_clips = []
        missing_folders = []
        for folder_name in TRAIN_FOLDERS:
            folder_path = os.path.join(INPUT_ROOT, folder_name)
            clips = collect_clips(folder_path, folder_name)
            if clips:
                train_clips.extend(clips)
            else:
                missing_folders.append(folder_name)

        if missing_folders:
            print(f"[WARNING] These folders are missing or empty: {missing_folders}")

        overall_stats["train"] = process_split(
            train_clips, "TRAIN (F1-F34)", detector, video_processor
        )

    # Validation
    if "val" in args.splits:
        val_clips = collect_clips(os.path.join(INPUT_ROOT, VAL_FOLDER), VAL_FOLDER)
        overall_stats["val"] = process_split(
            val_clips, "VAL", detector, video_processor
        )

    # Test
    if "test" in args.splits:
        test_clips = collect_clips(os.path.join(INPUT_ROOT, TEST_FOLDER), TEST_FOLDER)
        overall_stats["test"] = process_split(
            test_clips, "TEST", detector, video_processor
        )

    # Final summary 
    print(f"\n{'='*60}")
    print("  Final Summary")
    print(f"{'='*60}")
    for split, stats in overall_stats.items():
        ok    = stats.get("ok", 0) + stats.get("already_done", 0)
        total = sum(stats.values())
        print(f"  {split:<8}: {ok:>6} / {total} succeeded")
    print(f"\n  Output saved to:\n  {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
