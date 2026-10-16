#!/usr/bin/env python3
"""Extract frames from one or more videos at a chosen sampling rate (default 3 FPS)."""
import argparse
import sys
from pathlib import Path
import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}

def collect_videos(inputs):
    videos, seen = [], set()
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            candidates = [path]
        else:
            print(f"[SKIP] Missing path or unsupported file: {path}")
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                videos.append(resolved)
    return videos

def safe_stem(path):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem).strip("_") or "video"

def extract_video(video_path, output_root, sample_fps, resize, quality):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return 0
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / source_fps if source_fps > 0 else 0
    if source_fps <= 0:
        print(f"[ERROR] Invalid FPS metadata: {video_path}")
        cap.release()
        return 0

    output_dir = output_root / safe_stem(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / sample_fps
    next_sample_time = 0.0
    saved = 0
    frame_index = 0

    print(f"\n[VIDEO] {video_path}")
    print(f"        Source FPS: {source_fps:.3f} | Duration: {duration:.1f}s")
    print(f"        Output: {output_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / source_fps
        if timestamp + (0.5 / source_fps) >= next_sample_time:
            if resize:
                frame = cv2.resize(frame, tuple(resize), interpolation=cv2.INTER_AREA)
            timestamp_ms = int(round(timestamp * 1000))
            name = f"{safe_stem(video_path)}_{timestamp_ms:010d}ms.jpg"
            if cv2.imwrite(str(output_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                saved += 1
                next_sample_time += interval
        frame_index += 1

    cap.release()
    print(f"        Saved {saved} images")
    return saved

def main():
    parser = argparse.ArgumentParser(description="Extract video frames at a fixed sampling rate.")
    parser.add_argument("inputs", nargs="+", help="Video file(s) and/or folder(s) containing videos.")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames to extract per second (default: 3).")
    parser.add_argument("--output", default="dataset/images", help="Output directory (default: ./dataset/images).")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), help="Optional resize, e.g. --resize 640 480.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 0-100 (default: 95).")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be between 0 and 100.")
    if args.resize and min(args.resize) <= 0:
        parser.error("--resize dimensions must be positive.")

    videos = collect_videos(args.inputs)
    if not videos:
        print("No supported videos found.")
        sys.exit(1)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total = sum(extract_video(v, output_root, args.fps, args.resize, args.quality) for v in videos)
    print(f"\n[DONE] Processed {len(videos)} video(s); saved {total} images.")
    print(f"[OUTPUT] {output_root}")

if __name__ == "__main__":
    main()
