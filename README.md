# Text-to-image retrieval with relevance feedback

This is an official implementation of the paper:
*"A Little More Like This: Text-to-Image Retrieval with Vision-Language Models Using Relevance Feedback"*

## Getting started

Suggested Python version: 3.11

CUDA version: 12.8

Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

We use [Weights and Biases](https://wandb.ai/) for logging and visualization. Please sign up for an account and run:
```
wandb login
```

If you do not want to use `wandb`:
* append `--disable_wandb` flag in the retrieval script: `src/retrieval_pipeline.py`. 
* remove wandb logger from the training script for AFS: `src/train_summarizer.py`.

## Datasets

### Download Flickr30k dataset and Karpathy splits:

1. Create a directory for the dataset:
```
mkdir data
mkdir data/flickr30k
```

2. Check Flickr Terms of Use and download the Flickr30k images ([here](https://hockenmaier.cs.illinois.edu/DenotationGraph/data/)) and unzip them to `data/flickr30k/flickr30k-images/` directory.

3. Download the annotations for Karpathy splits (e.g., [here](https://cs.stanford.edu/people/karpathy/deepimagesent/)) and unzip the JSON with annotations as `data/flickr30k/dataset_flickr30k.json`. 

You can use alternative file paths but make sure to update the data config files in `configs/flickr30k/data*.yaml`.

### Download COCO-2014 dataset and Karpathy splits:

Check COCO official website [here](https://cocodataset.org/#home). The following commands can be used to download the dataset and Karpathy splits:

```
mkdir data
mkdir data/coco

python -m src.datasets.download_coco_karpathy_splits --output_dir data/coco/annotations

cd data
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

unzip train2014.zip -d coco/
unzip val2014.zip -d coco/
```

### Download MSR-VTT dataset:

MSR-VTT is a short-video captioning/retrieval dataset. We use the standard "1k-A" split (9,000 train / 1,000 test videos, 20 captions per training video), matching the protocol used by CLIP4Clip and most video-text retrieval literature, plus a validation subset held out from the training videos (see below).

The original MSR-VTT release links are no longer reliable, so we use two third-party mirrors instead:
* Raw video clips (~6.1GB): hosted by the [Frozen-in-Time](https://github.com/m-bain/frozen-in-time) authors.
* Captions and 1k-A split files: from the [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) GitHub release.

```
mkdir data
mkdir data/msrvtt

python -m src.datasets.download_msrvtt_splits --output_dir data/msrvtt
```

This downloads and extracts both archives, and writes converted annotations to `data/msrvtt/annotations/{train,val,test}.json`, in the same per-line JSON format used by the COCO/Flickr30k datasets above (`filepath`, `filename`, `sentences`, `sentids`, `imgid`).

Notes:
* The video archive is large (~6.1GB). If you'd rather fetch it yourself (e.g. with a resumable download manager), place it at `data/msrvtt/MSRVTT.zip` before running the command above -- the script detects the existing file and skips re-downloading it.
* Use `--skip_videos` to only download and convert the (small) annotation files, without pulling the large video archive -- useful for checking the setup before committing to the full download.
* Test videos are paired with exactly one caption each (the official JSFusion evaluation caption), matching the standard retrieval protocol; train/val videos keep all 20 captions each.
* The 1k-A protocol only defines train (9,000) and test (1,000) videos, with nothing held out for validation. We carve `--num_val_videos` (default 500) videos out of the training set instead, using a seeded shuffle (`--seed`, default 28) for reproducibility. This validation split is only used for early-stopping/checkpoint selection during AFS training -- it is not part of the reported retrieval benchmark.
* Frames are sampled with TSN-style segment sampling (`num_frames` in the data config, default 12): the clip is split into that many segments and one frame is picked per segment -- a random frame within the segment while training (cheap temporal augmentation), the segment's center frame at eval time (deterministic and reproducible). `segment_overlap` (default `0.0`, range `[0, 1)`) widens each segment beyond its non-overlapping width so neighboring segments share part of their frame range; see `configs/msrvtt/data_overlap.yaml` for an example. Note this only changes anything at *train* time (it shifts which frame is picked deterministically at eval, but doesn't add sampling diversity there, since eval never uses randomness).

You can use an alternative `--output_dir`, but make sure to update the data config files in `configs/msrvtt/data*.yaml` accordingly.

### VATEX dataset (investigated, not currently used)

We investigated [VATEX](https://eric-xw.github.io/vatex-website/index.html) as a second video-text retrieval dataset alongside MSR-VTT (mirroring how the original paper pairs COCO with Flickr30k), but decided to defer it. This is documented here so it isn't re-investigated from scratch later.

**Why VATEX looked like a good fit:** 10 independent English captions per video -- structurally compatible with this codebase's query/held-out-ground-truth design (one caption as query, the rest as ground truth for AFS/GRF), unlike alternatives such as DiDeMo or ActivityNet Captions, whose standard retrieval protocol concatenates all of a video's descriptions into a single paragraph (effectively 1 caption/video, which does not work with that design).

**The blocker: no direct raw-video download exists for VATEX.** Unlike MSR-VTT (whose raw clips are redistributed directly via a third-party mirror), no comparable mirror hosts VATEX's ~29k train+validation video clips. The official VATEX site only distributes precomputed features, not raw video (see [this unanswered GitHub issue asking exactly this](https://github.com/CASIA-IVA-Lab/VALOR/issues/31)). Captions/annotations themselves are readily available (`HuggingFaceM4/vatex` via the `datasets` library gives `train`/`validation`/`public_test`/`private_test` splits with `videoID`, `start`, `end`, `enCap`), but the actual video pixels require scraping YouTube directly using those ids and timestamps.

**What we built and verified before hitting the blocker:** [download_vatex_splits.py](src/datasets/download_vatex_splits.py) uses `yt-dlp` (with `--download-sections` to fetch only the needed clip range, forcing an H.264/mp4 format selection since YouTube's default AV1/webm output isn't decodable by `decord`) against the HF annotation source above. This was verified working correctly end-to-end at small scale -- real downloads (5 and then 500-per-split smoke tests), including a post-download `decord`-readability check per clip and a val/test carve-out from the official validation split (1,500/1,500, matching the HGR retrieval-eval protocol used in the literature) via `--seed`.

**Why it's deferred:** at the scale needed for the full dataset (~29,000 clips), concurrent `yt-dlp` requests against YouTube triggered bot detection ("sign in to confirm you're not a bot"). This can only be reliably worked around with proxies or a much slower/patient scraping schedule (low concurrency, spread over days), neither of which felt proportionate just to acquire a secondary dataset.

**Alternatives considered and also rejected:**
* **DiDeMo**: caption structure doesn't fit (paragraph-per-video eval protocol, not multiple independent captions), and its Flickr/YFCC100M video source has its own access problems (Yahoo Webscope, YFCC100M's original distribution channel, no longer offers access).
* **ActivityNet Captions**: avoids YouTube scraping (official request-form access to a Google/Baidu Drive-hosted mirror of real video files), but has the same paragraph-per-video caption structure problem as DiDeMo, plus a much larger footprint (long-form videos, ~849 video-hours total vs. VATEX's ~10s clips) and a time-boxed 7-day access window.
* **LSMDC**: avoids scraping (direct download after a license agreement), but is a different content domain (movie clips with audio-description narration) and its caption-per-clip structure (likely one aligned description per clip, not several independent ones) was not verified in detail before deprioritizing it.

**Current status:** proceeding with MSR-VTT only. `download_vatex_splits.py` and its dependencies (`yt-dlp`, `imageio-ffmpeg` in `requirements.txt`) remain in the repo in case this is revisited later with a slower scraping schedule (e.g. on infrastructure where a multi-day low-intensity background job is practical).

## Scripts

Below are the example commands for running and training different components. You can find more bash scripts for different models and datasets in `scripts/` directory.

### Retrieval pipeline

The main retrieval inference script is available in `src/retrieval_pipeline.py`. It can be used to run the retrieval with and without relevance feedback:
* To run the retrieval pipeline with GRF, you need to generate captions with LLaVA using `src/captioning_pipeline.py` (instructions below).
* To run the retrieval pipeline with AFS, you need to generate embeddings and retrieval results with `src/run_embeddings_and_retrieval.py` and then train the summarizer model with `src/train_summarizer.py` (instructions below).

Example script for retrieval with explicit feedback:
```
python -m src.retrieval_pipeline \
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation gt_user \
```
<details>
<summary>Arguments</summary>

- `--dataset`: Dataset to use (options: coco, flickr).
- `--data_config`: Path to the data config file.
- `--experiment_config`: Path to the experiment config file with summarizer parameters (optional).
- `--model_family`: Model family to use (e.g., blip2).
- `--model_id`: Model ID to use (optional).
- `--batch_size`: Batch size for DataLoader, default is 4.
- `--split`: Dataset split to use (train, val, test), default is test.
- `--device`: Device to use for inference (cuda or cpu), defaults to cuda if available.
- `--debug`: Enables debug mode, processing only the first 10 batches.
- `--use_8bit`: Enables 8-bit quantization.
- `--load_embeddings`: Path to load embeddings from a file (optional).
- `--save_embeddings`: Path to save embeddings to a file (optional).
- `--summarizer_checkpoint`: Path to the summarizer checkpoint file (optional).
- `--text_from`: Text field used for retrieval, default is text_class_label.
- `--no_metrics`: Do not report metrics.
- `--no_plots`: Do not plot metrics.
- `--num_turns`: Number of turns to run, default is 1.
- `--feedback_aggregation`: Feedback aggregation method (options: images, object_detection), default is images.
- `--top_k_feedback`: Number of top k images to feedback, default is 5.
- `--top_k_eval`: Number of top k images to evaluate, default is [1, 5].
- `--temperature`: Temperature for softmax, default is 0.01

</details>

Example script for zero-shot retrieval on video (MSR-VTT), no feedback:
```
python -m src.retrieval_pipeline \
    --dataset msrvtt \
    --data_config configs/msrvtt/data.yaml \
    --model_family clip_video \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 8 \
    --split test \
    --text_from caption_0 \
    --no_plots \
    --disable_wandb
```

`clip_video` ([src/models/clip_video.py](src/models/clip_video.py)) adapts a frozen image CLIP model to video: each sampled frame is encoded independently through the same vision tower a single-image CLIP model would use, and frame embeddings are mean-pooled into one video embedding (the "MeanP" approach from CLIP4Clip). Text is encoded completely independently of any video -- unlike native video-text models such as X-CLIP, whose text embeddings are cross-attended against a specific video's features and are therefore not precomputable ahead of retrieval, which would be incompatible with this codebase's "encode everything once, compare via one similarity matrix" design. On the MSR-VTT test split this gets hits@1/5/10 of 30.6%/53.6%/62.5% (MRR@10 40.3%), matching published zero-shot CLIP mean-pooling baselines for this benchmark.

### Generate embeddings and retrieval results

In order to train the attentive feedback summarizer (AFS), we need to generate embeddings and retrieval results. The script for this is available in `src/run_embeddings_and_retrieval.py`.

Example script:
```
python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 8 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/blip2/test \
    --split test;
```

This also supports `--dataset msrvtt` with `--model_family clip_video` (or any other registered model family), the same way as the retrieval pipeline above.

### LLaVa for caption generation

Example command for caption generation using LLaVa: `llava-hf/llava-1.5-7b-hf`:
```
python -m src.captioning_pipeline \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --model_family llava \
    --model_id llava-hf/llava-1.5-7b-hf \
    --batch_size 8 \
    --split test \
    --use_8bit \
    --by_image_path
```

The custom bash script for caption generation is available in `scripts/run_captioning_pipeline.sh`.

### Summarizer training
It requires generated embeddings and retrieval results along with the captions generated by LLaVa.

Example script:
```
python -m src.train_summarizer \
    --data_config configs/flickr30k/data_summarizer_clip.yaml \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --experiment_config configs/flickr30k/clip_local_summarizer.yaml \
    --num_workers 16
```

Check bash scripts `scripts/train_summarizer_clip.sh`, `scripts/train_summarizer_clip_large.sh` and `scripts/train_summarizer_blip2.sh` for more examples.

### Video backbone fine-tuning

Unlike the paper's original design (a frozen pretrained VLM backbone, with only the AFS summarizer trained on top), the video backbone can optionally be fine-tuned directly. This is video-specific: CLIP was never trained on video, so its zero-shot transfer to averaged-frame video representations benefits far more from fine-tuning than the equivalent gap does for still images -- e.g. CLIP4Clip reports zero-shot CLIP mean-pooling at ~27% R@1 on MSR-VTT vs. ~43% after fine-tuning the exact same architecture. The script for this is available in `src/train_backbone.py`.

Two modes, controlled by `--freeze_backbone`:
* **Full fine-tuning** (default): every parameter of the underlying CLIP model (vision tower, text tower, both projection layers) is trainable. Adapts the model most thoroughly, at the highest compute cost and the highest risk of overfitting/forgetting on a comparatively small dataset like MSR-VTT's ~8.5k training videos.
* **Partial fine-tuning** (`--freeze_backbone`): the vision and text towers are frozen exactly as pretrained; only the final projection layers are trainable. Much cheaper and lower-risk, at the cost of a more limited adaptation.

Both use a standard symmetric contrastive loss -- CLIP's own training objective: within a batch, matching (video, caption) pairs are pulled together and every other pairing in the batch is pushed apart (`ContrastiveLoss` in [src/models/clip_video_finetuner.py](src/models/clip_video_finetuner.py)). This differs from `CosineSimilarityLoss` ([src/models/attentive_summarizer.py](src/models/attentive_summarizer.py)), which only pulls positive pairs together and is used for training the AFS summarizer on top of a frozen backbone, not for fine-tuning the backbone itself.

Example script:
```
python -m src.train_backbone \
    --data_config configs/msrvtt/data.yaml \
    --model_family clip_video \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32 \
    --max_epochs 10 \
    --num_workers 4
```

Add `--freeze_backbone` for partial fine-tuning instead. `--debug` runs 2 train/val batches only, for verifying the training loop is wired correctly before committing to a full run. Fine-tuning the full CLIP model (~151M parameters) is impractical on CPU -- a GPU is effectively required.