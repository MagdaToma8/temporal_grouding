"""
One-off analysis script: splits MSR-VTT test-set retrieval performance into
"temporal-dependent" vs "static-sufficient" caption categories (labels produced
by manual/LLM classification, see analysis/msrvtt_test_temporal_labels.json),
to answer whether temporal information matters more for one category than the other.

Not part of the regular pipeline -- mirrors the embedding/retrieval logic in
src/retrieval_pipeline.py but keeps per-query ranks instead of only aggregate metrics.

Example:
    python -m src.analyze_temporal_split \\
        --data_config configs/msrvtt/data.yaml \\
        --labels_file analysis/msrvtt_test_temporal_labels.json \\
        --label zero-shot_12frame
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.msrvtt import load_msrvtt_data
from src.models.clip_video_finetuner import load_finetuned_clip_state_dict
from src.models.configs import get_model_config
from src.utils.utils import load_yaml_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_config", required=True)
    p.add_argument("--backbone_checkpoint", default=None)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label", default="run", help="Name for this run, printed in the output")
    return p.parse_args()


def main():
    args = parse_args()
    data_config = load_yaml_file(args.data_config)

    model_config = get_model_config("clip_video", "openai/clip-vit-base-patch32")
    model = model_config["model_class"].from_pretrained(model_config["model_id"], trust_remote_code=True)
    if args.backbone_checkpoint:
        model.load_state_dict(load_finetuned_clip_state_dict(args.backbone_checkpoint))
        print(f"Loaded fine-tuned backbone weights from {args.backbone_checkpoint}")
    model = model.to(args.device)
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)

    dataset, collator = load_msrvtt_data(data_config, "test", processor, process_images=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    image_embeddings, text_embeddings, img_paths = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            attention_mask = batch["caption_0_attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            outputs = vlm_wrapper.get_embeddings(inputs={
                "pixel_values": batch["image"].to(args.device),
                "input_ids": batch["caption_0"].to(args.device),
                "attention_mask": attention_mask,
            })
            image_embeddings.append(outputs["image_embeds"].detach().cpu())
            text_embeddings.append(outputs["text_embeds"].detach().cpu())
            img_paths.extend(batch["img_path"].tolist())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # logits_per_text[i, j] = similarity of query i's text to video j's image embedding
    logits_per_text = torch.matmul(text_embeddings, image_embeddings.t())
    sorted_indices = torch.argsort(logits_per_text, dim=1, descending=True)  # [num_queries, num_videos]

    # Each video has exactly one caption in the test split, iterated in the same order for both
    # text and image embeddings, so query i's correct match is video i itself.
    num_queries = sorted_indices.shape[0]
    ranks = torch.empty(num_queries, dtype=torch.long)
    for i in range(num_queries):
        # position (0-indexed) of the correct video's index within the sorted ranking
        ranks[i] = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1

    labels = json.load(open(args.labels_file))
    categories = np.array([
        labels.get(os.path.basename(p), "unknown") for p in img_paths
    ])

    def report(mask, name):
        r = ranks[mask].float()
        n = len(r)
        if n == 0:
            print(f"{name}: no examples")
            return
        hits1 = (r <= 1).float().mean().item()
        hits5 = (r <= 5).float().mean().item()
        hits10 = (r <= 10).float().mean().item()
        mrr10 = torch.where(r <= 10, 1.0 / r, torch.zeros_like(r)).mean().item()
        print(f"[{args.label}] {name} (n={n}): hits@1={hits1:.4f} hits@5={hits5:.4f} hits@10={hits10:.4f} mrr@10={mrr10:.4f}")

    report(np.ones(num_queries, dtype=bool), "overall")
    report(categories == "temporal", "temporal")
    report(categories == "static", "static")
    unknown_count = int((categories == "unknown").sum())
    if unknown_count:
        print(f"[{args.label}] WARNING: {unknown_count} videos had no label match (filename mismatch?)")


if __name__ == "__main__":
    main()
