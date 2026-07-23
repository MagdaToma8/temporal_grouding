import argparse
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
import torch
import decord
from datasets import load_dataset
from tqdm import tqdm


# HuggingFaceM4/vatex mirrors the official VATEX annotations (videoID, start/end
# timestamps, English + Chinese captions) as a clean tabular dataset, rather than
# requiring us to parse the "{id}_{start}_{end}" combined videoID string ourselves
# the way the original VATEX release JSON does. Loading it runs the repo's own
# dataset-loading script (trust_remote_code=True) -- HuggingFaceM4 is a well-known,
# widely-used HF org, so this is a reasonable trust call, but it's worth being
# explicit that this executes third-party code rather than just parsing data.
ANNOTATIONS_REPO = "HuggingFaceM4/vatex"

# YouTube frequently serves AV1/webm by default, which decord's bundled ffmpeg build
# cannot decode (verified: it throws a pixel-format filter-graph error). Forcing an
# H.264-in-mp4 selection (falling back to any mp4, then to whatever's available)
# keeps output videos in a format decord can actually read, matching MSR-VTT's videos.
FORMAT_SELECTOR = "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def _clip_id(video_id: str, start: int, end: int) -> str:
    # a handful of VATEX videoIDs repeat with different (start, end) segments
    # (same source video, different clipped moments) -- so videoID alone is not a
    # unique filename; include the timestamps, matching the original VATEX convention.
    return f"{video_id}_{start:06d}_{end:06d}"


def _download_clip(video_id: str, start: int, end: int, out_path: str, ffmpeg_path: str):
    if os.path.exists(out_path):
        return True, None

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--ffmpeg-location", ffmpeg_path,
        "-f", FORMAT_SELECTOR,
        "--remux-video", "mp4",
        "--retries", "3",
        "--no-warnings",
        "--quiet",
        "-o", out_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "yt-dlp failed with no stderr output"
        return False, error

    # yt-dlp/ffmpeg succeeding doesn't guarantee decord can actually decode the
    # result (e.g. a video with no avc1 track at all falls back to a codec decord
    # can't handle) -- verify now, while we're already paying the download cost,
    # rather than discovering it silently much later during training.
    try:
        video_reader = decord.VideoReader(out_path)
        if len(video_reader) == 0:
            raise RuntimeError("0 frames decoded")
    except Exception as e:
        os.remove(out_path)
        return False, f"downloaded but unreadable by decord: {e}"

    return True, None


def _download_split(entries, videos_dir: str, ffmpeg_path: str, num_workers: int):
    os.makedirs(videos_dir, exist_ok=True)
    successes = []
    failures = []

    def _task(entry):
        clip_id = _clip_id(entry["videoID"], entry["start"], entry["end"])
        out_path = os.path.join(videos_dir, f"{clip_id}.mp4")
        ok, error = _download_clip(entry["videoID"], entry["start"], entry["end"], out_path, ffmpeg_path)
        return entry, clip_id, ok, error

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_task, entry) for entry in entries]
        for future in tqdm(as_completed(futures), total=len(futures)):
            entry, clip_id, ok, error = future.result()
            if ok:
                successes.append((entry, clip_id))
            else:
                failures.append({
                    "videoID": entry["videoID"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "error": error,
                })

    return successes, failures


def _build_entries(successes, videos_filepath: str, start_imgid: int):
    entries = []
    for i, (item, clip_id) in enumerate(successes):
        entries.append({
            "filepath": videos_filepath,
            "filename": f"{clip_id}.mp4",
            "sentences": item["enCap"],
            "sentids": list(range(len(item["enCap"]))),
            "imgid": start_imgid + i,
        })
    return entries


def _write_jsonl(entries, out_path: str):
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main(output_dir: str, num_workers: int, seed: int, limit, splits_to_run):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    print("Loading VATEX annotations from Hugging Face...")
    ds = load_dataset(ANNOTATIONS_REPO, trust_remote_code=True)

    # Standard retrieval-eval protocol (HGR, Chen et al. 2020; reused by CLIP4Clip
    # and others): split the official 3,000-video validation set in half (1,500/1,500)
    # for val/test, and use the official 25,991-video train split as-is. We do NOT use
    # VATEX's own public_test/private_test splits -- those serve the VATEX captioning
    # challenge, not the retrieval benchmark protocol we're matching against.
    val_pool = list(ds["validation"])
    rng = random.Random(seed)
    rng.shuffle(val_pool)
    half = len(val_pool) // 2

    split_entries = {
        "train": list(ds["train"]),
        "val": val_pool[:half],
        "test": val_pool[half:],
    }

    if limit is not None:
        split_entries = {k: v[:limit] for k, v in split_entries.items()}

    videos_filepath = "videos"
    videos_dir = os.path.join(output_dir, videos_filepath)
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    next_imgid = 0
    for split in ["train", "val", "test"]:
        if splits_to_run and split not in splits_to_run:
            continue

        entries = split_entries[split]
        print(f"--- {split}: downloading {len(entries)} clips (num_workers={num_workers}) ---")
        successes, failures = _download_split(entries, videos_dir, ffmpeg_path, num_workers)
        print(f"{split}: {len(successes)} succeeded, {len(failures)} failed")

        if failures:
            failures_path = os.path.join(annotations_dir, f"{split}_failures.json")
            with open(failures_path, "w") as f:
                json.dump(failures, f, indent=2)
            print(f"Logged {len(failures)} failures to {failures_path}")

        jsonl_entries = _build_entries(successes, videos_filepath, next_imgid)
        next_imgid += len(jsonl_entries)
        _write_jsonl(jsonl_entries, os.path.join(annotations_dir, f"{split}.json"))
        print(f"Wrote {len(jsonl_entries)} entries to {annotations_dir}/{split}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the VATEX dataset via yt-dlp")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Concurrent yt-dlp downloads")
    parser.add_argument(
        "--seed", type=int, default=28,
        help="Seed for the val/test carve-out from the official validation set"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of videos per split -- useful for a quick smoke test before the full run"
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=None, choices=["train", "val", "test"],
        help="Only download these splits (default: all three)"
    )
    args = parser.parse_args()
    main(args.output_dir, args.num_workers, args.seed, args.limit, args.splits)
