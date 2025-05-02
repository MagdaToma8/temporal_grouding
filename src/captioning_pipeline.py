import os
import json
import argparse
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
    parser = argparse.ArgumentParser(description="Captioning pipeline for CUB dataset with LLaVA")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use (cub/flickr/coco)"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data config file"
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        help="Model family to use (e.g., llava)"
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
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--random_per_class",
        type=int,
        default=None,
        help="Random per class"
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="captions",
        help="Output file name"
    )
    parser.add_argument(
        "--by_image_path",
        action="store_true",
        default=False,
        help="Output by image path"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

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
    data_config = load_yaml_file(args.data_config)
    if args.dataset == "cub":
        dataset, dataset_collator = load_cub_data(data_config, args.split, processor, process_images=False)
    elif args.dataset == "flickr":
        dataset, dataset_collator = load_flickr_data(data_config, args.split, processor, process_images=False)
    elif args.dataset == "coco":
        dataset, dataset_collator = load_coco_data(data_config, args.split, processor, process_images=False)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset_collator
    )

    captions = []
    img_paths = []

    prompt_pool = [
        "Describe the image and main visual features one sentence.",
        "Generate a short caption for this image, focusing on the visual details.",
        "Provide a concise description, one sentence, of the image visual features and surroundings.",
        "Write a brief caption, one sentence, that describes the image visual features.",
        "Summarize the image visual features in a precise caption, maximum one sentence."
    ]

    for i, batch in enumerate(tqdm(dataloader)):
        images = batch['image']
        img_path_batch = batch['img_path']

        ## First turn

        prompts = [prompt_pool[j % len(prompt_pool)] for j in range(len(images))]

        processed_prompt = vlm_wrapper.process_inputs(
            image=images,
            prompt=prompts
        )

        outputs = vlm_wrapper.generate(processed_prompt)

        generated_text_all = vlm_wrapper.decode(outputs)
        generated_text = [text.split("ASSISTANT: ")[-1] for text in generated_text_all]
        print(generated_text)

        ## Second turn
        generated_text_all = [text.replace("USER:  \n", "USER: <image>\n") for text in generated_text_all]
        refining_prompt = "</s>USER: Pay attention to the visual settings and details on the image. Write exactly one sentence under 10 words. ASSISTANT:"
        refined_prompts = [generated_text_all[i] + refining_prompt for i in range(len(images))]
        print(img_path_batch)
        print(refined_prompts)
        processed_prompt = vlm_wrapper.process_inputs(
            apply_template=False,
            image=images,
            prompt=refined_prompts,
        )

        outputs = vlm_wrapper.generate(processed_prompt)
        generated_text_all = vlm_wrapper.decode(outputs)
        generated_text = [text.split("ASSISTANT: ")[-1] for text in generated_text_all]

        print(generated_text)
        captions.extend(generated_text)
        img_paths.extend(img_path_batch)

        if args.debug and i > 10:
            break

    # Save captions and image paths
    output_dir = os.path.join(data_config["data_dir"], "captions")
    os.makedirs(output_dir, exist_ok=True)

    if args.by_image_path:
        results = {os.path.basename(img_path): caption for img_path, caption in zip(img_paths, captions)}
    else:
        results = {
            'captions': captions,
            'img_paths': img_paths
        }

    with open(os.path.join(output_dir, f"{args.output_file_name}_{args.split}.json"), 'w') as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
