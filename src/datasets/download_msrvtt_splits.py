import argparse
import csv
import glob
import json
import os
import random
import zipfile
from urllib.request import urlretrieve

from tqdm import tqdm

# downloads both zips, and extracts them 
ANNOTATIONS_URL = "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip"
VIDEOS_URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"


def _download(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Already downloaded: {dest_path}")
        return

    print(f"Downloading {url} -> {dest_path}")
    with tqdm(unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
        def _hook(block_num, block_size, total_size):
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urlretrieve(url, dest_path, reporthook=_hook)


def _video_id_to_int(video_id: str) -> int:
    # video ids look like "video1234"
    return int(video_id.replace("video", ""))


def _load_captions_by_video(data_json_path: str):
    with open(data_json_path, "r") as f:
        data = json.load(f)
    captions_by_video = {}
    for item in data["sentences"]:
        captions_by_video.setdefault(item["video_id"], []).append(item["caption"])
    return captions_by_video


def _read_train_9k_ids(csv_path: str):
    with open(csv_path, "r") as f:
        return [row["video_id"] for row in csv.DictReader(f)]


def _read_jsfusion_test(csv_path: str):
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows or "sentence" not in rows[0]:
        raise ValueError(
            f"Expected a 'sentence' column in {csv_path}, found columns: "
            f"{list(rows[0].keys()) if rows else 'none'}. "
            "The JSFusion test csv format may have changed -- inspect the file manually."
        )
    video_ids = [row["video_id"] for row in rows]
    caption_by_video = {row["video_id"]: row["sentence"] for row in rows}
    return video_ids, caption_by_video


def _locate_videos_dir(output_dir: str) -> str:
    matches = glob.glob(os.path.join(output_dir, "**", "videos", "all"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            "Could not find a 'videos/all' directory after extracting MSRVTT.zip. "
            "The archive's internal folder structure may differ from what this script expects -- "
            "check the extracted contents manually."
        )
    return matches[0]


def _locate_file(output_dir: str, filename: str) -> str:
    matches = glob.glob(os.path.join(output_dir, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under {output_dir} after extracting msrvtt_data.zip. "
            "The archive's internal folder structure may have changed -- check the extracted contents manually."
        )
    return matches[0]


def _build_split_entries(video_ids, captions_by_video, filepath, single_caption_per_video=None):
    entries = []
    missing = []
    for video_id in video_ids:
        if single_caption_per_video is not None:
            sentences = [single_caption_per_video[video_id]]
        elif video_id in captions_by_video:
            sentences = captions_by_video[video_id]
        else:
            missing.append(video_id)
            continue
        entries.append({                    # as in COCODataset
            "filepath": filepath,
            "filename": f"{video_id}.mp4",
            "sentences": sentences,
            "sentids": list(range(len(sentences))),
            "imgid": _video_id_to_int(video_id),
        })
    if missing:
        print(f"Warning: {len(missing)} video ids had no captions in MSRVTT_data.json and were skipped")
    return entries


def _write_jsonl(entries, out_path: str):
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main(output_dir: str, num_val_videos: int, seed: int, skip_videos: bool):
    os.makedirs(output_dir, exist_ok=True)

    annotations_zip = os.path.join(output_dir, "msrvtt_data.zip")
    _download(ANNOTATIONS_URL, annotations_zip)
    with zipfile.ZipFile(annotations_zip, "r") as zf:
        zf.extractall(output_dir)

    videos_filepath = None
    if not skip_videos:
        videos_zip = os.path.join(output_dir, "MSRVTT.zip")
        _download(VIDEOS_URL, videos_zip)
        print("Extracting videos (this is a large archive, may take a while)...")
        with zipfile.ZipFile(videos_zip, "r") as zf:
            zf.extractall(output_dir)
        videos_dir = _locate_videos_dir(output_dir)
        # store as a forward-slash relative path (not os.path.relpath's native separator):
        videos_filepath = os.path.relpath(videos_dir, output_dir).replace(os.sep, "/")
        num_videos_found = len(glob.glob(os.path.join(videos_dir, "*.mp4")))
        print(f"Found {num_videos_found} video files under {videos_filepath}")
    else:
        # Assume the standard frozen-in-time layout even if videos aren't downloaded yet,
        # so annotations can still be generated ahead of time.
        videos_filepath = "videos/all"

    captions_by_video = _load_captions_by_video(_locate_file(output_dir, "MSRVTT_data.json"))
    train_9k_ids = _read_train_9k_ids(_locate_file(output_dir, "MSRVTT_train.9k.csv"))
    test_1k_ids, test_1k_captions = _read_jsfusion_test(_locate_file(output_dir, "MSRVTT_JSFUSION_test.csv"))

    assert len(train_9k_ids) == 9000, f"Expected 9000 train_9k video ids, found {len(train_9k_ids)}"
    assert len(test_1k_ids) == 1000, f"Expected 1000 JSFusion test video ids, found {len(test_1k_ids)}"

    # The standard 1k-A protocol uses all 10k videos for train_9k + test_1k, leaving
    # nothing for validation. Carve a val subset out of train_9k instead: used only
    # for early-stopping/checkpoint selection during AFS training, never reported
    # as a retrieval benchmark number.
    rng = random.Random(seed)
    shuffled_train_ids = train_9k_ids[:]
    rng.shuffle(shuffled_train_ids)
    val_ids = sorted(shuffled_train_ids[:num_val_videos])
    train_ids = sorted(shuffled_train_ids[num_val_videos:])

    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    _write_jsonl(
        _build_split_entries(train_ids, captions_by_video, videos_filepath),
        os.path.join(annotations_dir, "train.json")
    )
    _write_jsonl(
        _build_split_entries(val_ids, captions_by_video, videos_filepath),
        os.path.join(annotations_dir, "val.json")
    )
    _write_jsonl(
        _build_split_entries(
            test_1k_ids, captions_by_video, videos_filepath,
            single_caption_per_video=test_1k_captions
        ),
        os.path.join(annotations_dir, "test.json")
    )

    print(
        f"Wrote {len(train_ids)} train / {len(val_ids)} val / {len(test_1k_ids)} test "
        f"video entries to {annotations_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the MSR-VTT dataset (1k-A split)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument(
        "--num_val_videos",
        type=int,
        default=500,
        help="Number of videos held out from the 9k train split for validation"
    )
    parser.add_argument("--seed", type=int, default=28, help="Random seed for the train/val split")
    parser.add_argument(
        "--skip_videos",
        action="store_true",
        default=False,
        help="Skip downloading the (large) raw video archive; only fetch/convert annotations"
    )
    args = parser.parse_args()
    main(args.output_dir, args.num_val_videos, args.seed, args.skip_videos)
