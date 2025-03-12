import argparse
import os

from datasets import load_dataset


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("yerevann/coco-karpathy")
    ds["train"].to_json(os.path.join(output_dir, "train.json"))
    ds["validation"].to_json(os.path.join(output_dir, "val.json"))
    ds["test"].to_json(os.path.join(output_dir, "test.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save COCO Karpathy splits")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the dataset")
    args = parser.parse_args()
    main(args.output_dir)
