import os
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.datasets.cub import load_cub_data
from src.datasets.flickr import load_flickr_data
from src.datasets.coco import load_coco_data
from src.models.configs import get_model_config
from src.utils.quantization import bitsandbytes_8bit_config
from src.utils.utils import load_yaml_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieval evaluation pipeline for CUB dataset with BLIP2")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use (cub/flickr)"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data config file"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Path to the embeddings directory"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=10,
        help="Number of top results to save"
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        help="Model family to use (e.g., blip2)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        help="Model ID to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (train/val/test)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: take first 10 batches"
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--text_from",
        type=str,
        default="text_class_label",
        help="Text field used for retrieval"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for similarity computation"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    data_config = load_yaml_file(args.data_config)

    # Load model and processor
    model_config = get_model_config(args.model_family, args.model_id)
    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        quantization_config=bitsandbytes_8bit_config() if args.use_8bit else None
    )
    model = model.to(args.device) if not args.use_8bit else model
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])

    # Initialize VLMWrapper
    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)

    # Initialize dataset and dataloader
    if args.dataset == "cub":
        dataset, dataset_collator = load_cub_data(data_config, args.split, processor)
    elif args.dataset == "flickr":
        dataset, dataset_collator = load_flickr_data(data_config, args.split, processor)
    elif args.dataset == "coco":
        dataset, dataset_collator = load_coco_data(data_config, args.split, processor)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset_collator,
        shuffle=False
    )

    image_embeddings = []
    img_paths = []
    img_ids = []
    text_embeddings = {}

    for i, batch in enumerate(tqdm(dataloader, "Computing embeddings:")):
        images = batch['image'].to(args.device)
        img_path_batch = batch['img_path']
        img_ids.extend(batch["class_label"].tolist())

        image_embeds = None
        text_embeds = None

        for j in range(dataset.num_captions_to_use):
            input_ids = batch[f"caption_{j}"].to(args.device)
            attention_mask = batch[f"caption_{j}_attention_mask"].to(args.device)
            outputs = vlm_wrapper.get_embeddings(inputs={
                'pixel_values': images,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            if image_embeds is not None:
                assert torch.allclose(image_embeds, outputs['image_embeds'].detach().cpu())
            if text_embeds is not None:
                assert not torch.allclose(text_embeds, outputs['text_embeds'].detach().cpu())
            image_embeds = outputs['image_embeds'].detach().cpu()
            text_embeds = outputs['text_embeds'].detach().cpu()

            if f"caption_{j}" not in text_embeddings:
                text_embeddings[f"caption_{j}"] = []
            text_embeddings[f"caption_{j}"].append(outputs['text_embeds'].detach().cpu())

        image_embeddings.append(image_embeds)
        img_paths.extend(img_path_batch.tolist())
        if args.debug and i > 10:
            break

    image_embeddings = torch.cat(image_embeddings, dim=0)
    img_paths = np.array(img_paths)
    img_ids = np.array(img_ids)

    assert np.unique(img_ids, return_counts=True)[1].max() == 1

    for key in text_embeddings.keys():
        text_embeddings[key] = torch.cat(text_embeddings[key], dim=0)

    img_embeddings_dict = {}
    for i, img_path in tqdm(enumerate(img_paths), "Arranging image embeddings:"):
        img_embeddings_dict[os.path.basename(img_path)] = {
            'image_embeds': image_embeddings[i],
            'img_id': img_ids[i]
        }

    torch.save(
        img_embeddings_dict,
        os.path.join(args.embeddings_dir, "image_embeddings.pt")
    )

    for key in tqdm(text_embeddings.keys(), "Retrieval per caption:"):
        if args.chunk_size is not None:
            logits_per_image_chunks = []

            for start_idx in range(0, image_embeddings.size(0), args.chunk_size):
                end_idx = min(start_idx + args.chunk_size, image_embeddings.size(0))
                image_chunk = image_embeddings[start_idx:end_idx]
                logits_chunk = torch.matmul(image_chunk, text_embeddings[key].t())
                if 'blip2' in args.model_family:
                    logits_chunk, _ = logits_chunk.max(dim=1) # [num_images, num_queries]
                logits_per_image_chunks.append(logits_chunk)
                print(logits_chunk.shape)

            # Concatenate all chunks to form the complete similarity matrix
            logits_per_image = torch.cat(logits_per_image_chunks, dim=0)
            print(logits_per_image.shape)
        else:
            logits_per_image = torch.matmul(image_embeddings, text_embeddings[key].t())

            # Shape of image embeddings in BLIP-2: [num_images, n_q, dim] -- Q-former returns image representation as n_q tokens
            # Select the most relevant image token (from n_q) for each query
            if 'blip2' in args.model_family:
                logits_per_image, _ = logits_per_image.max(dim=1) # [num_images, num_queries]
        logits_per_text = logits_per_image.t() # [num_queries, num_images]

        sorted_indices = torch.argsort(logits_per_text, dim=1, descending=True)
        top_k_indices = sorted_indices[:, :args.save_top_k]
        relevant_images = img_paths[top_k_indices]

        text_embeddings_and_retrieval = {}
        for i, img_path in tqdm(enumerate(img_paths)):
            text_embeddings_and_retrieval[os.path.basename(img_path)] = {
                "text_embeds": text_embeddings[key][i],
                "retrieval_results": [os.path.basename(img_path) for img_path in relevant_images[i]]
            }

        torch.save(
            text_embeddings_and_retrieval,
            os.path.join(args.embeddings_dir, f"{key}_text_embeds_and_retrieval.pt")
        )


if __name__ == "__main__":
    main()
