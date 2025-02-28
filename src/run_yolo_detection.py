import os
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from ultralytics import YOLO

from src.utils.utils import load_yaml_file
from src.datasets.cub import load_cub_data
from src.datasets.flickr import load_flickr_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use (cub/flickr)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save detection results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (train/val/test)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode: process first 10 images only"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for detection"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    data_config = load_yaml_file(args.data_config)

    model = YOLO(args.model_path)

    image_dir = os.path.join(data_config["data_dir"], "flickr30k-images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if args.debug:
        image_files = image_files[:10]

    # Initialize dataset and dataloader
    if args.dataset == "cub":
        dataset, dataset_collator = load_cub_data(
            config=data_config,
            split=args.split,
            processor=None
        )
    elif args.dataset == "flickr":
        data_config["process_images"] = False
        dataset, dataset_collator = load_flickr_data(
            config=data_config,
            split=args.split,
            processor=None,
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    dataset_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset_collator,
        shuffle=False
    )

    results_dict = {}
    results_verbose = {}
    for batch in tqdm(dataset_loader, desc=f"Running yolo {args.model_path}"):
        images = batch["image"]
        img_paths = batch["img_path"]

        results = model.predict(
            source=images,
            save=True,
            project=os.path.join(args.output_dir, args.model_path),
            conf=args.conf_threshold,
            iou=0.5,
            imgsz=640
        )

        for i, result in enumerate(results):
            curr_image = os.path.basename(img_paths[i])
            if "cls" in args.model_path:
                results_dict[curr_image] = []
                if result.probs is not None:
                    confidences = (
                        (result.probs.top5conf >= args.conf_threshold)
                        .nonzero()
                        .detach().cpu().numpy()
                    )
                    cls = np.array(result.probs.top5)[confidences]
                    text_cls = [result.names[int(cls)] for cls in cls]
                    text_string_verbose = " ".join(text_cls)
                    results_verbose[curr_image] = text_string_verbose.replace("_", " ")
                    conf = result.probs.top5conf.detach().cpu().numpy()[confidences]
                    for j in range(len(cls)):
                        results_dict[curr_image].append({
                            "cls": int(cls[j]),
                            "text_cls": text_cls[j],
                            "conf": float(conf[j])
                        })
            else:
                results_dict[curr_image] = []
                verbose_text = result.verbose()
                verbose_text = verbose_text.replace("persons", "people")
                results_verbose[curr_image] = result.verbose().replace("persons", "people")
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
                        results_dict[curr_image].append({
                            "cls": cls,
                            "text_cls": result.names[cls],
                            "conf": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

    with open(os.path.join(args.output_dir, f"{args.split}_yolo.json"), "w") as f:
        json.dump(results_dict, f)

    if results_verbose:
        with open(os.path.join(args.output_dir, f"{args.split}_yolo_text.json"), "w") as f:
            json.dump(results_verbose, f)


if __name__ == "__main__":
    main()
