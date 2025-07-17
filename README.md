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