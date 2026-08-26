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

**Evaluating a fine-tuned checkpoint.** Pass the resulting `.ckpt` file's path via `--backbone_checkpoint` to either `retrieval_pipeline.py` or `run_embeddings_and_retrieval.py`, instead of the pretrained weights `--model_id` would otherwise load:
```
python -m src.retrieval_pipeline \
    --dataset msrvtt \
    --data_config configs/msrvtt/data.yaml \
    --model_family clip_video \
    --model_id openai/clip-vit-base-patch32 \
    --backbone_checkpoint checkpoints/backbone/<run>/epoch=7-val_loss=0.5414.ckpt \
    --batch_size 8 \
    --split test \
    --text_from caption_0 \
    --no_plots \
    --disable_wandb
```
This loads just the underlying CLIP weights out of the Lightning checkpoint (`load_finetuned_clip_state_dict` in [src/models/clip_video_finetuner.py](src/models/clip_video_finetuner.py) -- it strips the `vlm_wrapper_model.` prefix and drops the loss's own learnable temperature, which isn't part of the CLIP model itself) into a fresh `CLIPModel`, so it can be evaluated exactly like the pretrained one.

**Result**: full fine-tuning (~8 epochs before early stopping, no hyperparameter search) improved MSR-VTT test-set retrieval substantially over the zero-shot baseline reported above:

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| hits@1 | 30.6% | 36.8% |
| hits@5 | 53.6% | 65.8% |
| hits@10 | 62.5% | 76.2% |
| MRR@10 | 40.3% | 49.2% |

This sits below CLIP4Clip's own fully-tuned published number (~43% hits@1) -- expected, since this was a single untuned run rather than a hyperparameter search, and the direction/magnitude of improvement (not an exact match to their recipe) is the important finding here.

**Does `segment_overlap` help once training is actually involved?** Earlier, testing `segment_overlap=0.5` against the *zero-shot* backbone came back flat (see the MSR-VTT dataset section above) -- expected, since with nothing being trained, overlap can only shift which frame gets picked deterministically at eval, not provide a richer variety for a model to actually learn from. Re-running the exact same fine-tuning recipe with `configs/msrvtt/data_overlap.yaml` (`segment_overlap=0.5`) instead of `configs/msrvtt/data.yaml` -- identical batch size, epochs, and evaluation protocol, so the comparison isolates only the training-time sampling difference -- gives a modest additional improvement on top of fine-tuning:

| Metric | Zero-shot | Fine-tuned (no overlap) | Fine-tuned (overlap=0.5) |
|---|---|---|---|
| hits@1 | 30.6% | 36.8% | 39.1% |
| hits@5 | 53.6% | 65.8% | 66.0% |
| hits@10 | 62.5% | 76.2% | 77.1% |
| MRR@10 | 40.3% | 49.2% | 50.6% |

Every metric is equal-or-better with overlap enabled during training (none regressed), with the clearest gains in hits@1 (+2.3 pp) and MRR@10 (+1.4 pp) over the no-overlap fine-tune. This is a single run per condition, not repeated across multiple seeds, so treat the *magnitude* as indicative rather than statistically confirmed -- but the direction is exactly what the original hypothesis predicted: overlap's benefit only shows up once training can actually exploit the richer per-epoch sampling variety.

### Does temporal information actually help?

Before moving to a tubelet-based backbone, we ran two experiments to check whether the current mean-pooling backbone is extracting any genuine temporal/motion signal, or whether it's succeeding purely on single-frame appearance cues.

**Test 1: num_frames=1 vs num_frames=12.** Same zero-shot and fine-tuned (no-overlap) checkpoints as above, re-evaluated with `configs/msrvtt/data_1frame.yaml` (`num_frames: 1`) instead of the usual 12-frame config:

| Metric | Zero-shot, 12f | Zero-shot, 1f | Fine-tuned, 12f | Fine-tuned, 1f |
|---|---|---|---|---|
| hits@1 | 30.6% | 23.6% | 36.8% | 26.9% |
| hits@5 | 53.6% | 43.6% | 65.8% | 52.3% |
| hits@10 | 62.5% | 53.5% | 76.2% | 62.7% |
| MRR@10 | 40.3% | 32.2% | 49.2% | 38.0% |

Both backbones lose substantial performance going from 12 frames to 1 (as expected -- a single random-ish frame is a noisier summary of a video than 12 averaged together), and fine-tuning increases that reliance on multiple frames rather than reducing it (hits@1 drop: -7.0 pp zero-shot vs. -9.9 pp fine-tuned). This alone doesn't prove *temporal/motion* understanding, though -- since `CLIPVideoWrapper` mean-pools frame embeddings (order-invariant), the gain from more frames could equally be explained by "more independent visual samples of the same scene reduce the chance of a single bad/uninformative frame," with no actual motion modeling involved.

**Test 2: splitting retrieval by caption category.** To isolate whether the *content* of the gain is temporal, all 1,000 MSR-VTT test captions were classified (manually, by direct LLM judgment against a fixed rubric -- not a keyword heuristic) into two buckets:
* **temporal-dependent** (139 captions, 14%): the caption centers on an action/event whose defining characteristic is motion or a state change that a single still frame would likely misrepresent or leave ambiguous -- e.g. *"a bus crashes into a car"*, *"man standing on the ledge of a very tall building jumps off"*, *"polar bear jumps into water then plays around while people watch"*.
* **static-sufficient** (861 captions, 86%): identifiable from one representative frame -- scenes, objects, categories, talking/interview footage, generic cooking/dancing/sports mentions, etc. -- e.g. *"a man is talking about business"*, *"a woman is playing piano"*, *"cartoon show for kids"*.

Not part of the regular pipeline, but kept in the repo for reproducibility: labels are in [analysis/msrvtt_test_temporal_labels.json](analysis/msrvtt_test_temporal_labels.json) (`{filename: "temporal"|"static"}` for all 1,000 test videos), and the retrieval numbers below come from [src/analyze_temporal_split.py](src/analyze_temporal_split.py), a one-off analysis script that mirrors `retrieval_pipeline.py`'s embedding/similarity logic but keeps per-query ranks instead of only aggregate metrics:
```
python -m src.analyze_temporal_split \
    --data_config configs/msrvtt/data.yaml \
    --backbone_checkpoint checkpoints/backbone/<run>/epoch=7-val_loss=0.5414.ckpt \
    --labels_file analysis/msrvtt_test_temporal_labels.json \
    --label finetuned_12frame
```
(omit `--backbone_checkpoint` for the zero-shot backbone; swap `--data_config` to `configs/msrvtt/data_1frame.yaml` for the 1-frame condition.)

| Setup | Category (n) | hits@1 | hits@5 | hits@10 | MRR@10 |
|---|---|---|---|---|---|
| Zero-shot, 12 frames | overall (1000) | 30.6% | 53.6% | 62.7% | 40.4% |
| | temporal (139) | 36.0% | 64.0% | 71.9% | 48.1% |
| | static (861) | 29.7% | 51.9% | 61.2% | 39.1% |
| Fine-tuned, 12 frames | overall (1000) | 36.8% | 65.8% | 76.2% | 49.2% |
| | temporal (139) | 48.2% | 71.9% | 82.0% | 59.3% |
| | static (861) | 35.0% | 64.8% | 75.3% | 47.6% |
| Zero-shot, 1 frame | overall (1000) | 23.6% | 43.6% | 53.5% | 32.2% |
| | temporal (139) | 32.4% | 57.6% | 66.9% | 43.1% |
| | static (861) | 22.2% | 41.4% | 51.3% | 30.5% |
| Fine-tuned, 1 frame | overall (1000) | 26.9% | 52.3% | 62.7% | 38.0% |
| | temporal (139) | 36.0% | 60.4% | 74.8% | 48.1% |
| | static (861) | 25.4% | 51.0% | 60.7% | 36.3% |

**Finding (counterintuitive at first glance): temporal-dependent captions retrieve *better* than static ones, in every single setup above.** This is not evidence that the model understands motion -- it's much more likely a caption-distinctiveness confound. MSR-VTT's static captions are highly repetitive across the 1,000-video test set (many near-duplicate captions like "a man is talking" / "a woman is talking about X"), which makes retrieval intrinsically harder for that bucket regardless of what the model can see, since there are many confusable competitors. A caption like "a bus crashes into a car" is rare and specific, so it's easy to retrieve correctly even from imperfect visual understanding, simply because there's little competition among the other 999 candidates.

The frame-count ablation, split by category, doesn't support genuine temporal modeling either: if the backbone were exploiting motion, going from 12 frames to 1 should hurt temporal-labeled captions *more* than static ones. It doesn't, consistently -- zero-shot hits@1 drops 36.0%→32.4% (-3.6 pp) for temporal vs. 29.7%→22.2% (-7.5 pp) for static (static drops *more*); fine-tuned hits@1 drops 48.2%→36.0% (-12.2 pp) for temporal vs. 35.0%→25.4% (-9.6 pp) for static (roughly comparable, temporal only marginally larger).

**Conclusion:** `CLIPVideoWrapper`'s mean-pooling is architecturally order-invariant, so it cannot represent motion direction or sequence no matter how many frames it's given -- the benefit of more frames (Test 1) looks like visual-sampling robustness, not temporal understanding, and is consistent with Test 2 showing no extra frame-count penalty specifically for motion-heavy captions. This motivates the planned move to a tubelet-based backbone (VideoMAE), which encodes short spatio-temporal patches directly instead of pooling independently-encoded frames -- re-running this same temporal/static split afterward is the real test of whether architectural motion modeling (not just more frames) closes the gap on the temporal-dependent bucket specifically.

**Caveats:** the temporal/static classification is a single LLM annotation pass against a fixed rubric, not independently verified or double-annotated against human labels, and the temporal bucket is small (n=139 out of 1,000), so category-level numbers carry real sampling noise. Treat this as a directional signal rather than a statistically confirmed result.

### Video backbone, take two: ViCLIP-B (tubelet/spatiotemporal attention)

Following directly from the conclusion above, `CLIPVideoWrapper`'s per-frame-then-mean-pool design was swapped for [ViCLIP-B](https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf) (`src/models/viclip.py`, registered as model family `viclip`), a genuinely motion-aware video-text backbone from the [InternVid](https://arxiv.org/pdf/2307.06942.pdf) project. Unlike mean-pooling, ViCLIP's vision tower feeds every frame's patches into the *same* self-attention layers at once (joint spatiotemporal attention), so a patch in one frame can directly attend to a patch in another frame -- the representation can depend on cross-frame relationships, not just an average of independent per-frame encodings. Its architecture is CLIP's own ViT-B/16 with plain spatial attention replaced by this joint spatiotemporal attention, initialized from CLIP and further contrastively pretrained on InternVid-10M video-text pairs -- so, like the CLIP backbone, it comes with a usable zero-shot starting point rather than needing an alignment trained from nothing. At 149.6M parameters it's directly comparable in scale to the CLIP ViT-B/32 backbone used throughout the rest of this project.

Text is still encoded fully independently of any video (a plain CLIP text transformer, not cross-attended against video features), preserving the "encode everything once, compare via one similarity matrix" retrieval design that ruled out X-CLIP earlier.

**Integration notes** (this was a meaningfully bigger lift than `CLIPVideoWrapper`, which only needed a `transformers.CLIPModel` + `CLIPProcessor`):
* ViCLIP-B has no HuggingFace `AutoProcessor` -- it preprocesses frames with plain ImageNet normalization (not CLIP's own mean/std) and tokenizes text with its own BPE tokenizer at a 32-token context length (not CLIP's 77). `ViCLIPProcessor` reimplements both by hand, matching the reference `demo.ipynb` numbers exactly.
* The model loads via `AutoModel.from_pretrained(..., trust_remote_code=True)` (downloads and runs the repo's own modeling code), which pulled in several previously-unneeded dependencies one `ImportError` at a time: `einops`, `timm`, `fvcore`, `ftfy`, `regex` (all added to `requirements.txt`).
* The upstream `config.json` ships a `tokenizer_path` of `"./bpe_simple_vocab_16e6.txt.gz"` -- a bare relative path that only resolves if the process's cwd happens to be inside HF's dynamic-module cache, which it never is here. `ViCLIPModelLoader` (`src/models/viclip.py`) works around this by patching the config to point at our own vendored copy of the identical vocab file (`src/models/viclip_assets/`) before constructing the model, so the generic `model_class.from_pretrained(model_id, trust_remote_code=True)` call in `retrieval_pipeline.py` etc. keeps working unmodified.
* `tests/test_viclip_wrapper.py` proves `ViCLIPProcessor`'s reimplemented preprocessing is numerically identical (`torch.allclose`, `atol=1e-4`/`1e-5`) to feeding the same real MSR-VTT video frames through ViCLIP's own reference preprocessing and inference code, plus a shape/normalization check through the full `load_msrvtt_data` + collator + wrapper path -- the same "prove it" pattern used for `CLIPVideoWrapper`.
* `configs/msrvtt/data_viclip.yaml` uses `num_frames: 8` (ViCLIP's own pretraining frame count) rather than the 12 used for CLIP mean-pooling.

**Result**: zero-shot ViCLIP-B substantially outperforms zero-shot CLIP mean-pooling on MSR-VTT test:

| Metric | CLIP mean-pool, zero-shot (12 frames) | ViCLIP-B, zero-shot (8 frames) |
|---|---|---|
| hits@1 | 30.6% | 37.4% |
| hits@5 | 53.6% | 59.6% |
| hits@10 | 62.5% | 71.9% |
| MRR@10 | 40.3% | 47.3% |

This is directionally expected (ViCLIP-B was actually pretrained on large-scale video-text pairs with genuine temporal attention, versus CLIP mean-pooling being an image model repurposed for video) but there is no published MSR-VTT retrieval number for ViCLIP-B to cross-check against here -- only Kinetics zero-shot action-recognition numbers were available from the source repo -- so treat this as "large, sensible, non-degenerate improvement" rather than a literature-verified exact match.

**Temporal-vs-static split, revisited.** `src/analyze_temporal_split.py` was generalized to accept `--model_family`/`--model_id` (instead of being CLIP-only) and re-run on zero-shot ViCLIP-B (`configs/msrvtt/data_viclip.yaml`, same 1,000-video test set and temporal/static labels as before):

| Category (n) | CLIP mean-pool, zero-shot (12f) | ViCLIP-B, zero-shot (8f) |
|---|---|---|
| overall (1000) | hits@1=30.6% | hits@1=37.4% |
| temporal (139) | hits@1=36.0% | hits@1=48.9% |
| static (861) | hits@1=29.7% | hits@1=35.5% |
| **temporal − static gap** | **+6.3 pp** | **+13.4 pp** |

The temporal/static gap more than doubles under ViCLIP-B. Encouraging, but on its own this doesn't prove genuine temporal modeling -- ViCLIP-B is simply a better model overall (it beats CLIP on *both* categories) and was pretrained partly on action-recognition data, which could just mean better vocabulary for action words rather than actually using cross-frame motion.

**The decisive check: does the gap depend on having multiple frames?** Re-ran the same split with `configs/msrvtt/data_viclip_1frame.yaml` (`num_frames: 1`) -- ViCLIP-B's temporal positional embedding has a built-in single-frame path (verified with a smoke test before committing to the full run), so this ablation is directly comparable to the CLIP mean-pooling one from the section above:

| | ViCLIP-B, 8 frames | ViCLIP-B, 1 frame | drop |
|---|---|---|---|
| overall hits@1 | 37.4% | 29.4% | −8.0 pp |
| temporal hits@1 (n=139) | 48.9% | 39.6% | **−9.4 pp** |
| static hits@1 (n=861) | 35.5% | 27.8% | −7.8 pp |
| temporal − static gap | +13.4 pp | +11.8 pp | narrows |

This is the first result in the whole project pointing the *expected* direction: with CLIP mean-pooling, dropping to 1 frame hurt **static** captions more than temporal ones (−7.5 pp vs −3.6 pp) -- the wrong direction for "the model is using motion." With ViCLIP-B, it's reversed: **temporal** captions lose more from fewer frames (−9.4 pp vs −7.8 pp), and the temporal/static gap shrinks when frames are taken away. Both are consistent with the joint spatiotemporal attention actually depending on multiple frames specifically for motion-dependent content, not just averaging away single-frame noise.

**How much to trust this:** the effect is real but modest (1.6 pp difference in frame-count sensitivity between the two categories) and the temporal bucket is small (n=139), so this is directional evidence, not a statistically "bulletproof" result. But it's a meaningfully different pattern than CLIP mean-pooling showed on the exact same captions and exact same evaluation protocol, in the direction genuine temporal modeling would predict.

<!-- ### Fine-tuning ViCLIP-B -->
<!-- 
`train_backbone.py` and `CLIPVideoFineTuner` (`src/models/clip_video_finetuner.py`) were already family-agnostic in most respects (both take `--model_family`/`--model_id`), but fine-tuning ViCLIP-B surfaced two real bugs, both caught and fixed before running the actual cluster job:
* `ViCLIPModelLoader.from_pretrained` now always forces `trust_remote_code=True` internally, rather than relying on the caller to pass it -- `retrieval_pipeline.py` does, but `train_backbone.py`'s generic `model_config["model_class"].from_pretrained(model_config["model_id"])` call doesn't, so this silently would have failed.
* ViCLIP's own `__init__` freezes its text encoder by default (`freeze_text=True`), unlike a freshly-loaded `CLIPModel` where every parameter already requires grad. Without an explicit fix, "full fine-tuning" (`freeze_backbone=False`, the default) would have silently left ViCLIP's 63.4M-parameter text tower frozen the whole time. `CLIPVideoFineTuner.__init__` now force-unfreezes every backbone parameter before applying `freeze_backbone`, so "full" actually means full for every backbone family.
* The `freeze_backbone=True` (partial fine-tuning) branch was also generalized: CLIP exposes `visual_projection`/`text_projection` as separate `nn.Linear` submodules, while ViCLIP's equivalents (`vision_encoder.proj`, `text_encoder.text_projection`) are raw `nn.Parameter` tensors with no `.parameters()` of their own, so the branch now checks which shape the backbone has and targets the right thing either way. -->
<!-- 
Both fixes were verified locally with `--debug` runs before the real job: full fine-tuning reported `149M/149M` trainable params, partial reported `655K/149M` (exactly `768×512` vision proj + `512×512` text proj -- confirms the right two parameters, nothing more or less). -->

**Cluster run Fine-tuned ViCLIP-B**: same recipe as the CLIP fine-tune (full fine-tuning, 15 max epochs, early stopping patience 5, `configs/msrvtt/data_viclip.yaml`), `--batch_size 16` instead of CLIP's 32 (ViCLIP attends jointly over all 8 frames' patches at once -- 1,576 tokens per video vs. CLIP's independently-processed 50 tokens/frame -- so it needs more memory per example). Ran the full 15 epochs without early stopping triggering (val/loss was still improving at the end, reaching 0.290) -- more epochs might have helped further, left as a future refinement.

**Result**: fine-tuned ViCLIP-B is the best backbone in this project so far, and the temporal/static analysis was re-run on it too, at both frame counts:

| Setup | Category (n) | hits@1 | hits@5 | hits@10 | MRR@10 |
|---|---|---|---|---|---|
| Zero-shot, 8 frames | overall (1000) | 37.4% | 59.6% | 71.9% | 47.3% |
| | temporal (139) | 48.9% | 73.4% | 84.2% | 60.2% |
| | static (861) | 35.5% | 57.4% | 69.9% | 45.2% |
| **Fine-tuned, 8 frames** | **overall (1000)** | **44.7%** | **69.9%** | **79.6%** | **55.7%** |
| | temporal (139) | 54.0% | 78.4% | 87.8% | 64.7% |
| | static (861) | 43.2% | 68.5% | 78.3% | 54.3% |
| Zero-shot, 1 frame | overall (1000) | 29.4% | 50.3% | 59.6% | 38.0% |
| | temporal (139) | 39.6% | 65.5% | 74.8% | 49.8% |
| | static (861) | 27.8% | 47.9% | 57.1% | 36.1% |
| Fine-tuned, 1 frame | overall (1000) | 31.3% | 55.2% | 67.3% | 41.5% |
| | temporal (139) | 39.6% | 64.8% | 76.3% | 50.2% |
| | static (861) | 30.0% | 53.7% | 65.9% | 40.1% |

**Headline**: fine-tuned ViCLIP-B (44.7% hits@1) beats fine-tuned CLIP mean-pooling (36.8% hits@1, see above) by +7.9 pp -- the best result across every backbone tried in this project. Fine-tuning helped ViCLIP-B by roughly the same margin it helped CLIP (+7.3 pp vs. CLIP's +6.2 pp).

**Three more findings from re-running the frame-count/category analysis on the fine-tuned checkpoint:**
1. **Fine-tuning increased frame-count dependence, same as it did for CLIP.** Zero-shot ViCLIP-B drops 8.0 pp going from 8→1 frame; fine-tuned drops 13.4 pp -- mirrors CLIP's own zero-shot-vs-fine-tuned pattern (−7.0 pp vs −9.9 pp). Fine-tuning makes both architectures lean harder on having multiple frames available.
2. **The "temporal captions are more frame-sensitive" signal survives fine-tuning, at roughly the same modest size.** Fine-tuned: temporal drops 14.4 pp (8→1 frame) vs. static's 13.2 pp -- still temporal-drops-more, same direction as zero-shot's 9.4 pp vs. 7.8 pp. Fine-tuning neither erased nor dramatically amplified this signal.
3. **But fine-tuning's raw improvement was actually larger for static captions than temporal ones** -- static gained +7.7 pp (35.5%→43.2%) at 8 frames vs. temporal's +5.0 pp (48.9%→54.0%). If fine-tuning were specifically sharpening motion understanding, the opposite would be expected. This is consistent with the same caption-frequency explanation raised earlier: static captions are 86% of the training data, so there's simply more signal to learn from for that category -- fine-tuning's gains look like general improvement, not motion-targeted improvement.

**Overall takeaway across both experiments (zero-shot and fine-tuned):** ViCLIP-B's joint spatiotemporal attention shows real, if modest, evidence of using cross-frame information specifically for motion-dependent captions (finding 2, both zero-shot and fine-tuned) -- something CLIP mean-pooling never showed in either regime. But the *size* of the improvement from fine-tuning is not concentrated on temporal content (finding 3) -- fine-tuning mostly just makes the model better overall, on top of an architecture that was already somewhat motion-aware from pretraining. Both effects are small relative to the temporal bucket's sample size (n=139), so treat the direction as the finding, not the exact magnitude.

### Relevance feedback on video: PRF

The original paper's core contribution is relevance feedback on top of retrieval, not retrieval alone -- so with a working video backbone in hand, the next step was bringing that back in. `retrieval_pipeline.py`'s PRF path (`--feedback_aggregation images`) needed no code changes for video: it operates purely on already-computed embeddings under the `image_embeds` key, which `ViCLIPWrapper` already populates regardless of whether "image" means a photo or a video.

Tested on zero-shot ViCLIP-B, MSR-VTT test (1,000 videos), one feedback turn, top-5 pseudo-relevant items per query:

| Setting | hits@1 | hits@5 | hits@10 | MRR@10 |
|---|---|---|---|---|
| No feedback (baseline) | 37.4% | 59.6% | 71.9% | 47.3% |
| PRF, `rocchio_beta=0.1` (default) | 37.5% | 60.6% | 71.9% | 47.3% |
| PRF, `rocchio_beta=0.2` | 36.8% | 59.5% | 71.1% | 46.7% |

The default (conservative, 10%-weight) feedback gives a small, mostly-flat-to-positive nudge. Doubling the feedback weight makes things *worse* across almost every metric, not better. This makes sense given the mechanism: PRF has no ground truth, it just trusts its own top-5 retrieved results as "relevant." At 37% hits@1, a meaningful share of those top-5 lists are wrong, so weighting them more heavily amplifies whatever bias already exists in a middling initial retrieval rather than correcting it. A small nudge stays useful; a strong one doesn't.

### Toward AFS on video

AFS needs, per training example: a query caption, held-out ground-truth captions from the same item, and a first-pass retrieval's top-k results (loaded, as pixels, to feed the summarizer network). That last part already exists as `run_embeddings_and_retrieval.py`, which -- like PRF -- needed no code changes to work for video, verified both structurally (`--debug`) and by content (a query video's own basename showing up in its own retrieval results at a rate far above chance).

Ran at full scale on the fine-tuned ViCLIP-B checkpoint, on MSR-VTT `train` (8,500 videos) and `val` (500 videos), producing the embeddings + top-10 retrieval results AFS training will need. Self-retrieval rate on `train`/`caption_0` (500-video sample): 57.2% in top-10, 25.8% at rank 1 -- lower than the 79.6%/44.7% seen on the 1,000-video test set, but expected rather than a bug: the train candidate pool is 8.5x larger (much harder discrimination task), and `caption_0` specifically likely wasn't the caption fine-tuning happened to sample for that video in any given epoch (training resampled a random single caption per video every epoch, not a fixed one). Both self-retrieval rates are thousands of times above chance level (~0.01%/0.12% for a random guess among 8,500 candidates), confirming the embeddings/retrieval pipeline itself is correct.

Unlike PRF and the embeddings script, AFS training did **not** work out of the box: `train_summarizer.py` only had `flickr`/`coco` branches, and the summarizer-mode dataset (`FlickrDatasetSummarizer`) loads single images, not video frames. Now built:

* **`MSRVTTDatasetSummarizer`** (`src/datasets/msrvtt.py`) -- mirrors `FlickrDatasetSummarizer`: holds out one of a video's 20 captions as the query, keeps the rest as ground truth, and loads sampled frames for both the query video and its top-k retrieved neighbors (from the embeddings/retrieval results above), not just single images. Wired into `load_msrvtt_data(..., summarizer=True)`.
* **A video-aware mode in `SummarizerDatasetCollator`.** The query video batches as `[bsz, num_frames, C, H, W]`. Retrieved neighbors batch as `[bsz*topk, num_frames, C, H, W]` -- flat over `bsz*topk`, *not* reshaped to `[bsz, topk, ...]`, matching how the existing image-only path already works (`AttentiveSummarizer` regroups the resulting *embeddings* into `[bsz, topk, dim]` itself afterward, not the raw pixels). Caught by tracing through exactly how `AttentiveSummarizer` consumes the data, not just checking shapes looked plausible.
* **A real pre-existing bug fix in `AttentiveSummarizer` itself.** CLIP-style wrappers always compute vision and text outputs together in one call, so `_get_text_features` builds a throwaway dummy `pixel_values` tensor just to satisfy that -- shaped for a single image (`[bsz, 3, H, W]`), which crashes against any video wrapper (needs 5D). Added a `video_num_frames` constructor option (mirroring the existing `img_size` pattern) so the dummy is shaped correctly for video backbones too.
* An `msrvtt` branch in `train_summarizer.py`.

Verified with a local smoke test (real MSR-VTT videos, a small fake embeddings/retrieval-results directory): dataset/collator shapes checked first, then the full `AttentiveSummarizer.forward()` pass run end-to-end with real data (using `CLIPVideoWrapper` as a fast stand-in backbone, since this was only testing shape plumbing, not backbone-specific numerics). Both passed.

**Two more bugs surfaced once this ran for real on the cluster** (neither caught by the local smoke test above, since that used `CLIPVideoWrapper` as a fast stand-in and never exercised these codepaths):
* `AttentiveSummarizer`'s dummy *text* input (built the same way as the dummy pixel_values above, to satisfy CLIP-style wrappers that always compute both text and vision together) was a fixed, arbitrary length of 10 tokens. CLIP's text encoder tolerates any length up to 77; ViCLIP's does not -- it adds a fixed `[32, dim]` positional embedding directly with no slicing, so anything other than exactly 32 tokens crashes. Added a `text_seq_len` constructor option (same pattern as `video_num_frames`), set to `VICLIP_CONTEXT_LENGTH` (now a public constant in `src/models/viclip.py`) when the model family is `viclip`.
* `retrieval_pipeline.py`'s AFS evaluation path had two spots that assumed generated captions always exist (true for the image datasets it was written for), neither respecting `--summarizer_no_captions`: an unconditional captions-file load at startup, and (only in the "global embeddings" branch actually used here -- the "local embeddings" branch already had this guard) an unconditional attempt to compute caption-based negative relevance inside the per-query feedback loop. Both fixed to skip cleanly when `--summarizer_no_captions` is set.

**Training**: global mode, `configs/msrvtt/viclip_global_summarizer.yaml` (embed_dim 512, 8 heads, depth 2, batch size 8 -- much smaller than the image-only configs' 512, since each example now loads 6 videos' worth of frames instead of 6 images), on the fine-tuned ViCLIP-B backbone (`--backbone_checkpoint`, since `AttentiveSummarizer` re-encodes retrieved items' raw pixels/text at training time, not just the pre-computed embeddings used to pick them -- this needed adding `--backbone_checkpoint` support to `train_summarizer.py`, which didn't have it). Val loss dropped steadily and realistically over 30 real epochs (0.504 -> 0.441, still improving when the epoch limit hit) -- a genuine learning curve, distinct from a `--debug` run's fast drop from memorizing 2 fixed batches (0.949 -> 0.621).

**Result**: a full three-way comparison, all on the same fine-tuned ViCLIP-B backbone and the same 1,000-video test set:

| Setting | hits@1 | hits@5 | hits@10 | MRR@10 |
|---|---|---|---|---|
| No feedback (baseline) | **44.7%** | 69.9% | 79.6% | 55.7% |
| PRF (naive averaging) | 44.5% | 68.6% | 79.5% | 55.5% |
| AFS (trained) | 41.5% | 68.8% | 78.7% | 53.2% |

AFS underperforms *both* the baseline and PRF. Notably, PRF itself flips from mildly helpful to mildly harmful once measured on the fine-tuned (rather than zero-shot) backbone: on zero-shot ViCLIP-B, PRF nudged hits@1 up (37.4% -> 37.5%) and hits@5 up more (59.6% -> 60.6%); on the fine-tuned backbone it's negative across every metric. Plausible read: a stronger backbone leaves less genuine error for blind self-trust to correct, and more room for it to blur an already-good query with noise from near-but-wrong neighbors.

**Why AFS specifically did worse, not just "didn't help":** one of the pipeline's own diagnostic counts is the clearest signal -- AFS's raw output *alone* (no blending with the original query at all) gets the correct video at rank 1 for only **18.9%** of queries, far below the plain caption alone (44.7%) or even PRF's raw averaged vector. That rules out "the 10% Rocchio blend was too weak to show AFS's benefit" -- the learned output itself is a substantially *worse* standalone retrieval signal than doing nothing, so blending in more of it (as a stronger-beta experiment would) should be expected to hurt further, not help, mirroring the pattern already seen with PRF at higher beta.

The likely cause: this configuration is the simplest possible version of AFS -- **global embeddings only** (one blurry summary vector per retrieved video, no fine-grained patch/token detail) and **no generated captions** (no video-captioning pipeline exists yet to produce them). Both are real simplifications relative to richer configurations the original paper's image-domain AFS could draw on. With only a single coarse vector per retrieved item to work with, the network may simply lack enough signal to learn *when* a retrieved video's evidence is trustworthy versus noise -- a plausible explanation for why it converges to a real, low-but-decreasing loss (it's learning to fit *something*) while still producing a poor retrieval signal in practice. This is a legitimate result, not a bug: relevance feedback's benefit (confirmed for images in the original paper) does not obviously transfer to video without richer inputs than were available here.

**Next**: local-mode AFS (see below) is the remaining deferred piece that could supply the richer signal global-mode AFS seems to be missing.

### GRF on video

GRF (Generative Relevance Feedback) is the same Rocchio-style mechanism as PRF -- average the top-k results, nudge the query toward that average -- but averaging *AI-generated caption embeddings* of the top-k retrieved videos instead of their raw visual embeddings. The bet: a sentence like "a green car races through the street" may be a cleaner, more distilled relevance signal than a blurry pooled visual vector.

**Video captioning pipeline.** `captioning_pipeline.py` had no MSR-VTT/video branch, and the only captioning model wired in (`LLaVaWrapper`, image-only) can't take a whole video. Built `LlavaNextVideoWrapper` (`src/models/llava_next_video.py`), wrapping [LLaVA-NeXT-Video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) -- native `transformers` support (no `trust_remote_code`, unlike ViCLIP), fed all `num_frames` sampled frames per video (not one representative frame, unlike the original plan) so it can describe motion/events a single frame can't. Verified structurally first (fake frames through the real processor, checking `pixel_values_videos` comes out as `[batch, 8, 3, 336, 336]` and the dataloader's un-processed batch shape matches) before any cluster run, then confirmed on a real `--debug` cluster run that captions actually describe actions/events across frames (e.g. *"a green sports car racing through the streets, narrowly avoiding collisions"*), not just a static scene. Ran at full scale on the 1,000-video test split.

**Two more "written for images, breaks on video" bugs, both in `get_embeddings_from_captions`** (used by GRF to embed the top-k captions) -- neither caught until the real GRF-on-ViCLIP run, since this exact function had never been exercised with a video wrapper before (earlier AFS runs all used `--summarizer_no_captions`):
* The function builds a dummy image input just to satisfy wrappers whose `get_embeddings` always expects vision+text together -- hardcoded as `torch.rand(N, 3, 224, 224)`, shaped and typed for CLIP-style processors. `ViCLIPProcessor` needs real PIL images (it calls `.convert("RGB")`, which a raw tensor doesn't have) and a 5D `[batch, num_frames, C, H, W]` shape (it's a video model). Added a `video_num_frames` parameter that builds a correctly-shaped dummy video instead when set.
* HF processors return a `BatchFeature` (has `.to(device)`); `ViCLIPProcessor.__call__` returns a plain `dict` (no HF `AutoProcessor` equivalent exists for ViCLIP -- see its docstring), which has no `.to()`. Fixed by moving tensors to device manually when `.to()` isn't available.

**Result**: same fine-tuned ViCLIP-B backbone, same 1,000-video test set, extending the earlier comparison:

| Setting | hits@1 | hits@5 | hits@10 | MRR@10 |
|---|---|---|---|---|
| No feedback (baseline) | **44.7%** | 69.9% | 79.6% | 55.7% |
| PRF (naive averaging) | 44.5% | 68.6% | 79.5% | 55.5% |
| AFS (trained, global) | 41.5% | 68.8% | 78.7% | 53.2% |
| GRF (AI captions) | 44.3% | 68.7% | 79.1% | 54.8% |

GRF is essentially flat/mildly negative -- in the same ballpark as PRF, and clearly less harmful than AFS. The same diagnostic used for AFS applies here too: GRF's feedback vector *alone* (ignoring the original query entirely) gets hits@1 of only 26.6% -- much worse than the plain caption (44.7%), confirming this is "a small amount of mostly-noise blended in," not a strong standalone signal being underweighted. Notably, GRF's standalone signal (26.6%) is meaningfully better than AFS's (18.9%) -- the AI-caption text does carry more usable signal than AFS's blurry pooled vector, even though neither is good enough alone to beat doing nothing. This is consistent with the working theory from the AFS section: richer, more specific input (real sentences vs. one coarse vector) helps somewhat, but on this backbone, no feedback mechanism tried so far beats simply trusting the original query.

### Local-mode AFS

`ViCLIPWrapper`'s `vision_model_output`/`text_model_output` used to just duplicate the pooled global embedding rather than exposing genuine per-patch/per-token detail. Fixed by reimplementing the relevant parts of ViCLIP's own forward pass directly in the wrapper (`_encode_vision_full`/`_encode_text_full` in `src/models/viclip.py`) -- calling its sub-layers (`conv1`, `transformer`, `ln_post`, `ln_final`, etc.) in the same order its vendored code does, computed once and reused for both the pooled (global) and full-sequence (local) outputs, so this doesn't cost an extra forward pass on top of the existing global-mode path. Local vision output keeps the class token plus every patch across every frame (`[batch, 1 + 196*8, 768]` for the 8-frame config used throughout this project); local text output keeps every token position (`[batch, 32, 512]`) -- both unprojected (pre `proj`/`text_projection`), matching how CLIP's own HF wrapper already exposes `last_hidden_state` for local-mode AFS on images.

**Verified correct, not just "runs without crashing":** the pooled `image_embeds`/`text_embeds` this refactor now derives from the same shared computation are bit-identical (`torch.allclose`, max diff `0.0`) to the original separate `encode_vision`/`text_encoder` calls, checked against the real model weights (not a stand-in). This matters because those pooled embeddings are what every previously-reported number in this README (backbone results, PRF, AFS-global, GRF) depends on -- confirming the refactor is a pure extension, not a silent change to results already reported above.

**No re-run of `run_embeddings_and_retrieval.py` needed.** AFS training re-encodes retrieved items from their raw frames at train time (`MSRVTTDatasetSummarizer.__getitem__` loads pixels, not saved embedding vectors) -- the precomputed embeddings are only used to pick *which* videos are the top-k neighbors, not as the features AFS actually trains on. Switching from global to local mode is purely a config change (`configs/msrvtt/viclip_local_summarizer.yaml`: `global_embeddings_vision/text: False`, `vision_dim: 768`, `text_dim: 512` -- the pre-projection widths, vs. global mode's projected 512/512).

**Open question before a full training run: compute cost.** Each retrieved video is now 1,569 vision tokens (vs. one pooled vector), and with `top_k_feedback=5` retrieved videos plus the query video, a single training example attends over roughly 9,400 vision tokens -- around 30x more than any local-mode config tried in the image domain so far (`configs/flickr30k/clip_local_summarizer.yaml`, single CLIP-B/32 images, batch_size 512). Started conservatively at `batch_size: 1`, to be raised once real GPU memory usage from a `--debug` cluster run (`job_debug_afs_local_viclip.slurm`) is known.