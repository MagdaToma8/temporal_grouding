import argparse
import json
import os
from typing import Dict, Any

import torch
from transformers import AddedToken
from transformers import BitsAndBytesConfig

from src.models.configs import get_model_config
from src.inference.vlm_inference import VLMInference
from src.utils.image_utils import load_image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video Language Model Inference")
    # Input arguments
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="What is in the picture?",
        help="User prompt for the model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/results.json",
        help="Path to the output JSON file"
    )

    # Model configuration
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["blip2", "blip2-embeddings", "blip2-matching"],
        default="blip2",
        help="Model family to use"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="Exact model id to use"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision to use"
    )

    # Inference parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=250,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        default=False,
        help="Use 8-bit quantization"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Batch size for inference"
    )
    return parser.parse_args()


def prepare_input(
        model_family: str,
        image,
        user_prompt: str,
        max_new_tokens: int
) -> Dict[str, Any]:
    if model_family in ["blip2", "blip2-embeddings", "blip2-matching"]:
        return {
            "prompt": user_prompt,
            "image": image,
            "max_new_tokens": max_new_tokens
        }
    else:
        raise ValueError(f"Unsupported model family: {model_family}")


def bitsandbytes_8bit_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


if __name__ == "__main__":
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check data for inference
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = []
    if os.path.isdir(args.image_path):
        for file in os.listdir(args.image_path):
            if file.lower().endswith(image_extensions):
                images.append(os.path.join(args.image_path, file))
        if not images:
            raise ValueError(f"No image files found in {args.image_path}")
    else:
        if not args.image_path.lower().endswith(image_extensions):
            raise ValueError(f"Invalid image file extension: {args.image_path}")
        images.append(args.image_path)

    print(f"Found {len(images)} image(s) to process")

    # init models
    model_family = args.model_family.lower()
    model_config = get_model_config(model_family)

    if not model_config:
        raise ValueError(f"Unsupported model family: {model_family}")

    if model_config["model_id"] is not None:
        model_config["model_id"] = args.model_id


    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        device_map={"": 0},
        revision=args.revision,
        quantization_config=bitsandbytes_8bit_config() if args.use_8bit else None
    )

    processor = model_config["processor_class"].from_pretrained(
        model_config["model_id"],
        revision=args.revision
    )

    if args.model_family in ["blip2"]:
        # Add special tokens to the BLIP2 processor: https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042
        processor.num_query_tokens = model.config.num_query_tokens
        image_token = AddedToken("<image>", normalized=False, special=True)
        processor.tokenizer.add_tokens([image_token], special_tokens=True)

        model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
        model.config.image_token_index = len(processor.tokenizer) - 1

    # init inference class
    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)
    inference = VLMInference(vlm_wrapper)

    results = {
        "metadata": {
            "model_family": model_family,
            "model_id": args.model_id,
            "max_new_tokens": args.max_new_tokens,
            "use_8bit": args.use_8bit
        },
        "results": []
    }

    # run inference for all images
    batch_size = len(images) if args.batch_size == -1 else args.batch_size
    for i in range(0, len(images), batch_size):
        batch_images = images[i: i + batch_size]
        batch_loaded_images = [load_image(image_path) for image_path in batch_images]

        input_data = prepare_input(
            model_family=model_family,
            image=batch_loaded_images,
            user_prompt=args.user_prompt,
            max_new_tokens=args.max_new_tokens
        )

        if model_config["model_type"] == "generation":
            input_data["prompt"] = [args.user_prompt] * len(batch_images)
            responses = inference.run_inference(input_data)
            for i, image_path in enumerate(batch_images):
                results["results"].append({
                    "image_path": image_path,
                    "prompt": input_data["prompt"][i],
                    "model_output": responses[i]
                })
        elif model_config["model_type"] == "embeddings":
            model_outputs = inference.get_embeddings(input_data)
            similarity_scores = model_outputs.logits_per_image
            for i, image_path in enumerate(batch_images):
                similarity_score = similarity_scores[i].item()
                results["results"].append({
                    "image_path": image_path,
                    "user_prompt": args.user_prompt,
                    "similarity_score": similarity_score
                })
        else:
            raise ValueError(f"Unsupported model type: {model_config['model_type']}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f)
