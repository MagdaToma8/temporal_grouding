import os
import argparse
import pickle
import random
from typing import Optional, List, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import wandb

from src.datasets.coco import load_coco_data
from src.datasets.cub import load_cub_data
from src.datasets.flickr import load_flickr_data
from src.models.vlm_wrapper import VLMWrapper
from src.models.configs import get_model_config
from src.models.attentive_summarizer import AttentiveSummarizer
from src.utils.metrics import RetrievalEvaluator, plot_metrics
from src.utils.utils import load_yaml_file, load_json_file, generate_experiment_id


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
        "--experiment_config",
        type=str,
        default=None,
        help="Path to the experiment config file with summarizer parameters"
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
        "--load_embeddings",
        type=str,
        default=None,
        help="Load embeddings from file"
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default=None,
        help="Save embeddings to file"
    )
    parser.add_argument(
        "--summarizer_checkpoint",
        type=str,
        default=None,
        help="Path to the summarizer checkpoint file"
    )
    parser.add_argument(
        "--text_from",
        type=str,
        default="text_class_label",
        help="Text field used for retrieval"
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        default=False,
        help="Do not report metrics"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Do not plot metrics"
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        default=1,
        help="Number of turns to run"
    )
    parser.add_argument(
        "--feedback_aggregation",
        type=str,
        default=None,
        help="Feedback aggregation method (images/object_detection)"
    )
    parser.add_argument(
        "--top_k_feedback",
        type=int,
        default=5,
        help="Number of top k images to feedback"
    )
    parser.add_argument(
        "--top_k_eval",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Number of top k images to evaluate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature for softmax"
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--wandb_log_all_turns",
        action="store_true",
        default=False,
        help="Log all turns"
    )
    parser.add_argument(
        "--visualize_attention_ratio",
        type=float,
        default=0.0,
        help="Visualize attention maps for the given ratio of images"
    )
    parser.add_argument(
        "--original_rocchio",
        action="store_true",
        default=False,
        help="Use original Rocchio's algorithm (without weighted aggregation)"
    )
    parser.add_argument(
        "--rocchio_alpha",
        type=float,
        default=0.8,
        help="Original query weight"
    )
    parser.add_argument(
        "--rocchio_beta",
        type=float,
        default=0.1,
        help="Positive feedback weight"
    )
    parser.add_argument(
        "--rocchio_gamma",
        type=float,
        default=0.1,
        help="Negative feedback weight"
    )
    parser.add_argument(
        "--accumulate_feedback",
        action="store_true",
        default=False,
        help="Adjust query embeddings across turns."
             "If False, original queries are fused with new feedback on each turn."
             "If True, adjusted queries are further fused with new feedback on each turn."
    )
    parser.add_argument(
        "--summarizer_no_captions",
        action="store_true",
        default=False,
        help="Use summarizer without generated captions"
    )
    return parser.parse_args()


def rocchio_update(
        query_embeddings: torch.Tensor,
        avg_relevance_vector: torch.Tensor,
        avg_non_relevance_vector: Optional[torch.Tensor] = None,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        norm_output: bool = True
):
    """
    Update the query embeddings using Rocchio's algorithm
        upd_q = alpha * q + beta * positive_feedback - gamma * negative_feedback

    Args:
        query_embedddings: initial query embeddings
        avg_relevance_vector: average relevance (positive feedback) vector
        avg_non_relevance_vector: average non-relevance (negative feedback) vector
        alpha: coefficient for initial query embeddings
        beta: coefficient for positive feedback
        gamma: coefficient for negative feedback
        norm_output: whether to normalize the output
    """
    # If negative feedback is not available, set its coefficient to zero
    if avg_non_relevance_vector is None:
        avg_non_relevance_vector = torch.zeros_like(avg_relevance_vector)
        gamma = 0.0
    assert query_embeddings.shape == avg_relevance_vector.shape
    updated_query_embeddings = (
        alpha * query_embeddings + \
        beta * avg_relevance_vector - \
        gamma * avg_non_relevance_vector
    )
    if norm_output:
        updated_query_embeddings = F.normalize(updated_query_embeddings, p=2, dim=-1)
    return updated_query_embeddings


def retrieval_query_image_embeddings(
        query_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        blip2: bool = False
) -> torch.Tensor:
    """
    Compute similarity matrix between queries and images

    Args:
        query_embeddings: initial query embeddings
        image_embeddings: image embeddings
        blip2: blip2-based retrieval is a bit different, because there are multiple tokens for each query

    Returns:
        logits_per_text: similarity matrix between queries and images
            rows: queries
            columns: images
    """
    # Compute similarity matrix between queries and images
    logits_per_image = torch.matmul(image_embeddings, query_embeddings.t())
    assert logits_per_image.max() <= 1.0 and logits_per_image.min() >= -1.0
    # Shape of image embeddings in BLIP-2: [num_images, n_q, dim] -- Q-former returns image representation as n_q tokens
    # Select the most relevant image token (from n_q) for each query
    if blip2:
        logits_per_image, _ = logits_per_image.max(dim=1) # [num_images, num_queries]

    logits_per_text = logits_per_image.t() # [num_queries, num_images]
    return logits_per_text


def get_embeddings_from_captions(
        captions: List[str],
        processor: AutoProcessor,
        vlm_wrapper: VLMWrapper,
        cpu: bool = False
) -> torch.Tensor:
    """
    Get text embeddings from captions

    Args:
        captions: list of captions
        processor: processor used for vlm_wrapper
        vlm_wrapper: vlm_wrapper
    """
    # Tokenization with processor used for vlm_wrapper
    tokenized_text = processor(
        images=torch.rand(len(captions), 3, 224, 224), # dummy image inputs
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    text_tokens = {k: v for k, v in tokenized_text.items() if k != 'pixel_values'} # remove dummy image inputs

    if cpu:
        vlm_wrapper.model = vlm_wrapper.model.to("cpu")
    else:
        tokenized_text = tokenized_text.to(vlm_wrapper.model.device)

    caption_embeddings = vlm_wrapper.get_embeddings(
        inputs=tokenized_text
    )

    return (
        caption_embeddings,
        text_tokens
    )


def softmax_weighted_aggregation(
        embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        temperature: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted aggregation of embeddings

    Args:
        embeddings: embeddings to aggregate [num_embeddings, dim]
        reference_embeddings: reference embeddings [1, dim]
        temperature: temperature for softmax
    """
    if reference_embeddings.ndim == 1:
        reference_embeddings = reference_embeddings.unsqueeze(0)

    similarity = F.softmax(
        torch.matmul(embeddings, reference_embeddings.t()) / temperature,
        dim=0
    )
    positive_aggregation_vector = torch.sum(embeddings * similarity, dim=0)
    negative_aggregation_vector = torch.sum(embeddings * (1 - similarity), dim=0)
    return positive_aggregation_vector, negative_aggregation_vector


def process_yolo_texts(
        object_detection_list: List[str],
        classification_list: List[str]
) -> Tuple[str, str]:
    """
    Process object detection and classification texts into a single string
    """
    objects = set()
    classified = set()
    for text in object_detection_list:
        entities = text.split(", ")
        objects.update(
            ''.join(c for c in entity if c.isalpha() or c.isspace()) for entity in entities
        )
    for text in classification_list:
        entities = text.split(", ")
        classified.update(
            ''.join(c for c in entity if c.isalpha() or c.isspace()) for entity in entities
        )
    classification_string = " ".join(classified)
    object_detection_string = " ".join(objects)
    yolo_string = object_detection_string + " " + classification_string
    yolo_string = ' '.join(yolo_string.split())
    yolo_string = yolo_string.replace("_", " ")
    yolo_words = list(set(yolo_string.split()))
    return yolo_words


def main():
    random.seed(28)
    args = parse_arguments()
    data_config = load_yaml_file(args.data_config)
    experiment_id = generate_experiment_id()

    project_suffix = ""
    if args.wandb_log_all_turns:
        project_suffix = "-log-all"
    if args.rocchio_alpha != 0.8 or args.rocchio_beta != 0.1 or args.rocchio_gamma != 0.1:
        project_suffix += "-ablation-rocchio"
    if not args.disable_wandb:
        wandb.init(
            project=f"multi-turn-retrieval{project_suffix}",
            config=args,
            name=f"{args.model_id.split('/')[-1]}-{args.feedback_aggregation}-{experiment_id}"
        )

    # Load model and processor
    model_config = get_model_config(args.model_family, args.model_id)
    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        trust_remote_code=True
    )
    model = model.to(args.device)
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
        collate_fn=dataset_collator
    )

    # Load additional data for feedback aggregation
    if args.feedback_aggregation == "yolo" and args.dataset == "flickr":
        od_json_file = os.path.join(data_config["data_dir"], "yolo", "object_detection", f"{args.split}_yolo_text.json")
        object_detection_texts = load_json_file(od_json_file)

        cls_json_file = os.path.join(data_config["data_dir"], "yolo", "classification", f"{args.split}_yolo_text.json")
        classification_texts = load_json_file(cls_json_file)

    if args.feedback_aggregation in ["generated_captions", "attentive_summarizer", "attentive_summarizer_with_gt"]:
        generated_captions_file = os.path.join(data_config["data_dir"], "captions", f"captions_{args.split}.json")
        generated_captions = load_json_file(generated_captions_file)

    avg_relevance_vector = None
    avg_non_relevance_vector = None

    # These feedback loops DO NOT require re-calculating textual embeddings
    # The feedback is applied in the feature space
    if args.feedback_aggregation is None:
        num_turns_dataloader = 1
        num_turns_feedback = 1
    elif args.feedback_aggregation in [
        "gt_user",
        "yolo",
        "images",
        "generated_captions",
        "attentive_summarizer",
        "attentive_summarizer_with_gt"
    ]:
        num_turns_dataloader = 1
        num_turns_feedback = args.num_turns
    # These feedback loops require re-calculating textual embeddings
    # The feedback is applied directly to query text
    # This is not the case for any of the feedback loops in the paper
    else:
        num_turns_dataloader = args.num_turns
        num_turns_feedback = 1

    # Pass through the dataloader to calculate embeddings and collect necessary data
    for turn in range(num_turns_dataloader):
        image_embeddings = []
        ground_truth_labels = []
        img_paths = []
        text_embeddings = []
        vision_model_outputs = []
        text_model_outputs = []
        captions_input_ids = []
        attn_for_visualization_indices = []

        # In case embeddings are already calculated, load them
        # This is only applicable if we don't need to re-calculate embeddings with feedback
        if args.load_embeddings and os.path.exists(args.load_embeddings) and num_turns_dataloader == 1:
            embeddings_dict = torch.load(args.load_embeddings, weights_only=False)
            image_embeddings = embeddings_dict.get('image_embeds', None)
            ground_truth_labels = embeddings_dict.get('ground_truth', None)
            img_paths = embeddings_dict.get('img_paths', None)
            text_embeddings = embeddings_dict.get('text_embeds', None)
        # Generate embeddings from scratch
        else:
            for i, batch in enumerate(tqdm(dataloader)):
                images = batch['image'].to(args.device)
                # class_label is the query ID or ground truth for retrieval evaluation
                class_label = batch['class_label']
                img_path_batch = batch['img_path']

                input_ids = batch[args.text_from].to(args.device)
                attention_mask = batch[f"{args.text_from}_attention_mask"].to(args.device)

                outputs = vlm_wrapper.get_embeddings(inputs={
                    'pixel_values': images,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

                image_embeddings.append(outputs['image_embeds'].detach().cpu())
                text_embeddings.append(outputs['text_embeds'].detach().cpu())
                vision_model_outputs.append(outputs['vision_model_output'].detach().cpu())
                text_model_outputs.append(outputs['text_model_output'].detach().cpu())
                captions_input_ids.extend(input_ids.detach().cpu().tolist())
                ground_truth_labels.extend(class_label.tolist())
                img_paths.extend(img_path_batch.tolist())
                if args.debug and i > 10:
                    break

            image_embeddings = torch.cat(image_embeddings, dim=0)
            ground_truth_labels = np.array(ground_truth_labels)
            img_paths = np.array(img_paths)
            text_embeddings = torch.cat(text_embeddings, dim=0)
            vision_model_outputs = torch.cat(vision_model_outputs, dim=0)
            text_model_outputs = [output[i] for output in text_model_outputs for i in range(len(output))]

            if args.save_embeddings:
                os.makedirs(os.path.dirname(args.save_embeddings), exist_ok=True)
                embeddings_dict = {
                    'image_embeds': image_embeddings,
                    'ground_truth': ground_truth_labels,
                    'img_paths': img_paths,
                    'text_embeds': text_embeddings
                }
                torch.save(
                    embeddings_dict,
                    args.save_embeddings + ".pt" if not args.save_embeddings.endswith(".pt") else args.save_embeddings
                )

        orig_text_embeddings = text_embeddings.clone()
        orig_image_embeddings = image_embeddings.clone()

        # Feedback loops without re-calculating embeddings (when the textual query is fixed)
        for feedback_turn in range(num_turns_feedback):
            if avg_relevance_vector is not None:
                text_embeddings = rocchio_update(
                    query_embeddings=text_embeddings if args.accumulate_feedback else orig_text_embeddings,
                    avg_relevance_vector=avg_relevance_vector,
                    avg_non_relevance_vector=avg_non_relevance_vector,
                    alpha=args.rocchio_alpha,
                    beta=args.rocchio_beta,
                    gamma=args.rocchio_gamma,
                    norm_output=True
                )

                # Quick evaluations the feedback quality using hits@1
                # relevance feedback vector and ground truth image embeddings
                if "blip2" not in args.model_family:
                    relevance_image = torch.matmul(avg_relevance_vector, image_embeddings.t())
                    print((relevance_image.diag() - relevance_image.max(dim=1)[0] == 0).sum())

                    # original text query and image embeddings
                    orig_text_image = torch.matmul(orig_text_embeddings, image_embeddings.t())
                    print((orig_text_image.diag() - orig_text_image.max(dim=1)[0] == 0).sum())

                    # updated text query (with feedback) and image embeddings
                    text_image = torch.matmul(text_embeddings, image_embeddings.t())
                    print((text_image.diag() - text_image.max(dim=1)[0] == 0).sum())

            # Retrieval logic:
            # Textual class labels are the queries, and the images are the candidates.

            # We want to find the top k images for each query.
            # The queries indices are ground truth labels in this case (image IDs)
            query_idx = ground_truth_labels

            logits_per_text = retrieval_query_image_embeddings(
                query_embeddings=text_embeddings,
                image_embeddings=orig_image_embeddings,
                blip2='blip2' in args.model_family
            )

            # Sort similarities (logits) for each query in descending order
            sorted_indices = torch.argsort(logits_per_text, dim=1, descending=True)
            sorted_img_paths = img_paths[sorted_indices]

            # Evaluation and plotting routine
            if not args.no_metrics:
                evaluator = RetrievalEvaluator(
                    top_k=args.top_k_eval,
                    metrics=['hits', 'mrr'],
                )
                metrics, metrics_per_class = evaluator.evaluate(sorted_indices, query_idx)
                if not args.disable_wandb:
                    if args.wandb_log_all_turns:
                        wandb.log({
                            "metrics": metrics,
                        })
                    elif feedback_turn == num_turns_feedback - 1:
                        wandb.log({
                            "metrics": metrics,
                        })

                for metric, value in sorted(metrics.items()):
                    print(f"{metric}: {value}")

                if not args.no_plots and metrics_per_class and metrics:
                    plot_metrics(
                        metrics=metrics,
                        metrics_per_class=metrics_per_class,
                        out_dir=os.path.join(
                            'results',
                            'plots',
                            os.path.basename(data_config.get('captions_file', 'captions')).split('.')[0],
                            args.model_id.split("/")[-1]),
                        out_filename=f'{args.text_from}'
                    )

                if args.visualize_attention_ratio != 0 and feedback_turn == num_turns_feedback - 1:
                    results_after_aggregation = []
                    sorted_image_paths = sorted_img_paths[:, :args.top_k_feedback + 5]
                    topk_image_embeddings = image_embeddings[sorted_indices[:, :args.top_k_feedback + 5]]
                    for idx in attn_for_visualization_indices:
                        similarities = torch.matmul(
                            topk_image_embeddings[idx],
                            text_embeddings[idx].unsqueeze(0).T
                        ).detach().cpu().numpy()
                        results_after_aggregation.append(
                            {
                                "img_path": img_paths[idx],
                                "topk_img_paths": sorted_image_paths[idx],
                                "similarities": similarities
                            }
                        )
                    with open(os.path.join(
                        'results',
                        'xattn',
                        os.path.basename(data_config.get('captions_file', 'captions')).split('.')[0],
                        args.model_id.split("/")[-1],
                        args.summarizer_checkpoint.split("/")[-2],
                        f"results_after_aggregation_turn_{feedback_turn + 1}.pkl"
                    ), 'wb') as f:
                        pickle.dump(results_after_aggregation, f)

            # Feedback aggregation logic
            if feedback_turn < num_turns_feedback - 1:
                topk_image_paths = sorted_img_paths[:, :args.top_k_feedback]

                # Original Rocchio uses the least relevant items to define negative feedback
                if args.original_rocchio:
                    lastk_image_paths = sorted_img_paths[:, -args.top_k_feedback:]

                # PRF: Aggregate image from top-k retrieved images
                if args.feedback_aggregation == "images":
                    if "blip2" in args.model_family:
                        image_embeddings = image_embeddings.mean(dim=1)
                    topk_indices = sorted_indices[:, :args.top_k_feedback]
                    topk_image_embeddings = image_embeddings[topk_indices]

                    # Check if indexing works as expected
                    for j, indices in enumerate(topk_indices):
                        assert torch.allclose(image_embeddings[indices], topk_image_embeddings[j])

                    if args.original_rocchio:
                        lastk_indices = sorted_indices[:, -args.top_k_feedback:]
                        lastk_image_embeddings = image_embeddings[lastk_indices]
                        avg_relevance_vector = topk_image_embeddings.mean(dim=1)
                        avg_non_relevance_vector = lastk_image_embeddings.mean(dim=1)
                    else:
                        avg_relevance_vector = torch.zeros_like(text_embeddings)
                        avg_non_relevance_vector = torch.zeros_like(text_embeddings)
                        avg_relevance_vector[j] = F.normalize(avg_relevance_vector[j], p=2, dim=-1)
                        avg_non_relevance_vector[j] = F.normalize(avg_non_relevance_vector[j], p=2, dim=-1)
                        # Iterate over each query and aggregate top-k rerieved image embeddings
                        for j in range(text_embeddings.shape[0]):
                            # Obtain softmax similarity weights for each image embedding in top-k for a given query
                            avg_relevance_vector[j], avg_non_relevance_vector[j] = softmax_weighted_aggregation(
                                embeddings=topk_image_embeddings[j],
                                reference_embeddings=text_embeddings[j].unsqueeze(0),
                                temperature=args.temperature
                            )
                            avg_relevance_vector[j] = F.normalize(avg_relevance_vector[j], p=2, dim=-1)
                            avg_non_relevance_vector[j] = F.normalize(avg_non_relevance_vector[j], p=2, dim=-1)

                # Aggregate information from YOLO-based object detection and image classification
                # This experiment shows degrading performance and was not included in the paper
                elif args.feedback_aggregation == "yolo":
                    # Select top-k image paths for each query
                    avg_relevance_vector = torch.zeros_like(text_embeddings)
                    avg_non_relevance_vector = torch.zeros_like(text_embeddings)
                    for j, img_paths_list in tqdm(enumerate(topk_image_paths), "Extracting object detection feedback"):
                        # Get top-k object detection and classification texts for each query
                        topk_object_detection_texts = [
                            object_detection_texts[os.path.basename(img_path)] for img_path in img_paths_list
                        ]
                        topk_classification_texts = [
                            classification_texts[os.path.basename(img_path)] for img_path in img_paths_list
                        ]

                        # Merge object detection and classification texts into a single string
                        yolo_words = process_yolo_texts(
                            object_detection_list=topk_object_detection_texts,
                            classification_list=topk_classification_texts
                        )
                        # Get embeddings from YOLO-based object detection and image classification
                        yolo_embeddings, _ = get_embeddings_from_captions(
                            captions=yolo_words,
                            processor=processor,
                            vlm_wrapper=vlm_wrapper,
                        )
                        yolo_embeddings = yolo_embeddings['text_embeds'].detach().cpu()

                        # Aggregate YOLO-based object detection and image classification
                        avg_relevance_vector[j], avg_non_relevance_vector[j] = softmax_weighted_aggregation(
                            embeddings=yolo_embeddings,
                            reference_embeddings=text_embeddings[j].unsqueeze(0),
                            temperature=args.temperature
                        )

                # GRF: Aggregate information from AI-generated captions
                elif args.feedback_aggregation == "generated_captions":
                    if args.original_rocchio:
                        topk_image_paths = np.concatenate([topk_image_paths, lastk_image_paths], axis=1)
                        print(topk_image_paths.shape)
                    avg_relevance_vector = torch.zeros_like(text_embeddings)
                    avg_non_relevance_vector = torch.zeros_like(text_embeddings)

                    for j, img_paths_list in enumerate(tqdm(topk_image_paths, "Feedback from AI-generated captions")):
                        captions = [generated_captions[os.path.basename(img_path)] for img_path in img_paths_list]

                        object_text_embeddings, tokenized_text = get_embeddings_from_captions(
                            captions=captions,
                            processor=processor,
                            vlm_wrapper=vlm_wrapper
                        )
                        object_text_embeddings = object_text_embeddings['text_embeds'].detach().cpu()

                        if args.original_rocchio:
                            avg_relevance_vector[j] = object_text_embeddings[:args.top_k_feedback].mean(dim=0)
                            avg_non_relevance_vector[j] = object_text_embeddings[-args.top_k_feedback:].mean(dim=0)
                        else:
                            avg_pos_output, avg_neg_output = softmax_weighted_aggregation(
                                embeddings=object_text_embeddings,
                                reference_embeddings=text_embeddings[j].unsqueeze(0),
                                temperature=args.temperature
                            )
                            avg_relevance_vector[j] = avg_pos_output
                            avg_non_relevance_vector[j] = avg_neg_output

                # AFS: Attentive feedback summarizer for feedback aggregation from PRF and GRF
                elif args.feedback_aggregation in ["attentive_summarizer", "attentive_summarizer_with_gt"]:
                    experiment_config = load_yaml_file(args.experiment_config)
                    assert experiment_config

                    # Initialize summarizer model
                    summarizer = AttentiveSummarizer(
                        pooler_config=experiment_config["pooler_config"],
                        text_dim_local=experiment_config.get("text_dim_local", experiment_config.get("text_dim", 768)),
                        text_dim_global=experiment_config.get("text_dim_global", experiment_config.get("text_dim", 768)),
                        vision_dim=experiment_config["vision_dim"],
                        vlm_wrapper=None,
                        global_embeddings_vision=experiment_config.get("global_embeddings_vision", True),
                        global_embeddings_text=experiment_config.get("global_embeddings_text", True),
                        checkpoint_path=args.summarizer_checkpoint
                    )
                    summarizer.eval()
                    # Select top-k retrieved image embeddings and paths for each query
                    topk_indices = sorted_indices[:, :args.top_k_feedback]
                    topk_image_embeddings = image_embeddings[topk_indices]
                    topk_vision_embeddings = vision_model_outputs[topk_indices]
                    topk_image_paths = sorted_img_paths[:, :args.top_k_feedback]

                    # Generate embeddings of AI-generated captions
                    captions_text_model_outputs = []
                    captions_embeddings = []
                    tokenized_texts = []
                    topk_captions = []

                    if args.feedback_aggregation == "attentive_summarizer_with_gt":
                        gt_captions = []
                        for j, batch in enumerate(tqdm(dataloader, "Getting ground truth captions")):
                            captions_keys = [f"caption_{c}" for c in range(dataset.num_captions_to_use)]
                            captions_keys = [cap for cap in captions_keys if cap != args.text_from]
                            captions_keys = random.sample(captions_keys, 1)
                            gt_caption_text = processor.tokenizer.batch_decode(
                                batch[captions_keys[0]],
                                skip_special_tokens=True
                            )
                            gt_captions.extend([caption.strip().capitalize() for caption in gt_caption_text])
                    if not args.summarizer_no_captions:
                        for i, img_paths_list in enumerate(
                            tqdm(topk_image_paths, "Getting embeddings of AI-generated captions")
                        ):
                            if args.feedback_aggregation == "attentive_summarizer_with_gt":
                                captions = [gt_captions[i]]
                            else:
                                captions = [generated_captions[os.path.basename(img_path)] for img_path in img_paths_list]
                            topk_captions.append(captions)
                            embeddings, tokenized_text = get_embeddings_from_captions(
                                captions=captions,
                                processor=processor,
                                vlm_wrapper=vlm_wrapper
                            )
                            captions_text_model_outputs.append(embeddings['text_model_output'].detach().cpu())
                            captions_embeddings.append(embeddings['text_embeds'].detach().cpu())
                            tokenized_texts.append(tokenized_text)
                        captions_embeddings = torch.stack(captions_embeddings, dim=0)

                    relevance_vector_neg = torch.zeros_like(text_embeddings)
                    relevance_vector_pos = torch.zeros_like(text_embeddings)
                    attn_for_visualization = []
                    attn_for_visualization_indices = []

                    num_text_items = 1 if (
                        args.feedback_aggregation == "attentive_summarizer_with_gt"
                    ) else args.top_k_feedback

                    for j in tqdm(
                        range(0, text_embeddings.shape[0]),
                        "Aggregating feedback using attentive summarizer"
                    ):
                        # Summarization of feedback using global embeddings
                        if (
                            experiment_config.get("global_embeddings_vision", True) and
                            experiment_config.get("global_embeddings_text", True)
                        ):
                            # Use summarizer to get query with feedback and cross-attention weights
                            #   summarized_vector will be used as positive relevance vector
                            #   xattn (cross-attention weights) will be used to compute negative relevance vector
                            with torch.no_grad():
                                summarized_vector, xattn = summarizer(
                                    text_embeddings[j].unsqueeze(0),
                                    captions_embeddings[j].unsqueeze(0) if not args.summarizer_no_captions else None,
                                    topk_image_embeddings[j].unsqueeze(0)
                                )

                            # Get cross-attention weights for image and text tokens
                            image_tokens_start = xattn.shape[-1] - topk_image_embeddings[j].shape[0]
                            xattn_text = xattn[..., :image_tokens_start]
                            xattn_image = xattn[..., image_tokens_start:]

                            # Sum cross-attention weights along attention heads and 2 output tokens
                            #   (CLS and text query embedding)
                            xattn_text_sum = (
                                xattn_text
                                .sum(dim=1)
                                .sum(dim=1)
                                .view(num_text_items, -1)
                            )
                            xattn_image_sum = (
                                xattn_image
                                .sum(dim=1)
                                .sum(dim=1)
                                .view(args.top_k_feedback, -1)
                            )

                            # Compute softmax of aggregated cross-attention weights
                            # (1 - xattn_text_sum) is used to get NEGATIVE relevance
                            xattn_text_softmax = F.softmax((1 - xattn_text_sum) / args.temperature, dim=0)
                            xattn_image_softmax = F.softmax((1 - xattn_image_sum) / args.temperature, dim=0)

                            # Aggregate relevance from each image and text embedding
                            neg_relevance_images = torch.sum(
                                topk_image_embeddings[j] * xattn_image_softmax,
                                dim=0
                            )
                            neg_relevance_texts = torch.sum(
                                captions_embeddings[j] * xattn_text_softmax,
                                dim=0
                            )

                        # Summarization of feedback using local embeddings (patches and tokens)
                        else:
                            # Use summarizer to get query with feedback and cross-attention weights
                            #   (from each patch and token)
                            with torch.no_grad():
                                summarized_vector, xattn = summarizer(
                                    text_model_outputs[j].unsqueeze(0),
                                    captions_text_model_outputs[j].unsqueeze(0) if (
                                        not args.summarizer_no_captions
                                    ) else None,
                                    topk_vision_embeddings[j].unsqueeze(0)
                                )
                            xattn = xattn.squeeze()

                            # Get cross-attention weights for image and text tokens
                            # In case of local embeddings:
                            #   number of text tokens is len_seq * args.top_k_feedback
                            #   number of image tokens is num_patches * args.top_k_feedback
                            image_tokens_start = xattn.shape[-1] - \
                                (topk_vision_embeddings[j].shape[1] * args.top_k_feedback)

                            xattn_text = xattn[..., :image_tokens_start] if not args.summarizer_no_captions else None
                            xattn_image = xattn[..., image_tokens_start:]

                            if random.random() < args.visualize_attention_ratio:
                                attn_for_visualization_indices.append(j)
                                query = processor.tokenizer.decode(captions_input_ids[j], skip_special_tokens=True)
                                img_path = img_paths[j]
                                topk_img_paths = topk_image_paths[j]
                                attn_per_patch = xattn_image.sum(dim=0).sum(dim=0).reshape(args.top_k_feedback, -1)

                                if not args.summarizer_no_captions:
                                    tokens = tokenized_texts[j]['input_ids'].flatten()
                                    tokens = processor.tokenizer.convert_ids_to_tokens(tokens)
                                    attn_per_token = xattn_text.sum(dim=0).sum(dim=0)
                                    attn_per_token_zip = list(zip(tokens, attn_per_token))
                                    attn_per_token_zip = [
                                        (x[0].replace("</w>", ""), x[1].item())
                                        for x in attn_per_token_zip
                                        if "</w>" in x[0]
                                    ]
                                similarities = torch.matmul(
                                    topk_image_embeddings[j],
                                    text_embeddings[j].unsqueeze(0).T
                                )
                                attn_for_visualization.append({
                                    "query": query,
                                    "img_path": img_path,
                                    "similarities": similarities,
                                    "topk_img_paths": topk_img_paths,
                                    "attn_per_patch": attn_per_patch,
                                    "attn_per_token": attn_per_token_zip
                                })

                            # Sum cross-attention weights along attention heads and output tokens
                            #   (<CLS>, query_token1, query_token2, ...)
                            if xattn_text is not None:
                                xattn_text_sum = (
                                    xattn_text.sum(dim=0).sum(dim=0)
                                    .reshape(num_text_items, -1)
                                    .sum(dim=1)
                                ).unsqueeze(0)

                                # Compute softmax of aggregated cross-attention weights
                                # -xattn_text_sum is used to get NEGATIVE relevance
                                xattn_text_softmax = F.softmax((-xattn_text_sum) / args.temperature, dim=0)
                                neg_relevance_texts = torch.sum(
                                    captions_embeddings[j] * xattn_text_softmax.unsqueeze(-1),
                                    dim=1
                                ).squeeze()
                            else:
                                neg_relevance_texts = None

                            xattn_image_sum = (
                                xattn_image.sum(dim=0).sum(dim=0)
                                .reshape(args.top_k_feedback, -1)
                                .sum(dim=1)
                            ).unsqueeze(0)

                            xattn_image_softmax = F.softmax((-xattn_image_sum) / args.temperature, dim=0)

                            # For BLIP-2, we use the max similarity
                            #   across the first dimension (learnable queries in Q-Former)
                            #   to get the negative relevance vector
                            if "blip2" in args.model_family:
                                logits_per_image = torch.matmul(
                                    topk_image_embeddings[j],
                                    text_embeddings[j].unsqueeze(0).T
                                )
                                logits_per_image, max_idx = logits_per_image.max(dim=1)
                                img_embeddings_max = topk_image_embeddings[
                                    j,
                                    torch.arange(topk_image_embeddings[j].shape[0]),
                                    max_idx.squeeze()
                                ]
                                neg_relevance_images = torch.sum(
                                    img_embeddings_max * xattn_image_softmax.unsqueeze(-1),
                                    dim=1
                                ).squeeze()
                            else:
                                neg_relevance_images = torch.sum(
                                    topk_image_embeddings[j] * xattn_image_softmax.unsqueeze(-1),
                                    dim=1
                                ).squeeze()

                        # Append positive relevance vector and average negative relevance vectors for texts and images
                        relevance_vector_pos[j] = summarized_vector
                        if neg_relevance_texts is not None:
                            relevance_vector_neg[j] = (neg_relevance_images + neg_relevance_texts) / 2
                        else:
                            relevance_vector_neg[j] = neg_relevance_images

                    avg_relevance_vector = relevance_vector_pos
                    avg_relevance_vector = F.normalize(avg_relevance_vector, p=2, dim=-1)

                    avg_non_relevance_vector = relevance_vector_neg
                    avg_non_relevance_vector = F.normalize(avg_non_relevance_vector, p=2, dim=-1)

                    if attn_for_visualization:
                        attn_save_path = os.path.join(
                            'results',
                            'xattn',
                            os.path.basename(data_config.get('captions_file', 'captions')).split('.')[0],
                            args.model_id.split("/")[-1],
                            args.summarizer_checkpoint.split("/")[-2]
                        )
                        os.makedirs(attn_save_path, exist_ok=True)
                        with open(os.path.join(attn_save_path, 'attn_for_visualization.pkl'), 'wb') as f:
                            pickle.dump(attn_for_visualization, f)

                # Explicit user feedback with additional ground truth captions
                elif args.feedback_aggregation == "gt_user":
                    avg_relevance_vector = torch.zeros_like(text_embeddings)
                    for j, batch in enumerate(tqdm(dataloader, "Explicit user feedback")):
                        captions_keys = [f"caption_{c}" for c in range(dataset.num_captions_to_use)]
                        captions_keys = [cap for cap in captions_keys if cap != args.text_from]

                        captions_keys = random.sample(captions_keys, 1)

                        # bsz, num_captions - 1, dim
                        gt_relevance = []

                        for caption in captions_keys:
                            input_ids = batch[caption].to(args.device)
                            attention_mask = batch[f"{caption}_attention_mask"].to(args.device)
                            # bsz, dim
                            outputs = vlm_wrapper.get_embeddings(inputs={
                                "pixel_values": torch.rand(args.batch_size, 3, 224, 224).to(args.device),
                                "input_ids": input_ids,
                                "attention_mask": attention_mask
                            })
                            gt_relevance.append(outputs['text_embeds'].detach().cpu())
                        gt_relevance = torch.stack(gt_relevance, dim=0).mean(dim=0)
                        avg_relevance_vector[args.batch_size * j: args.batch_size * (j + 1)] = gt_relevance
                    avg_relevance_vector = F.normalize(avg_relevance_vector, p=2, dim=-1)

                else:
                    raise ValueError(f"Invalid feedback aggregation method: {args.feedback_aggregation}. "
                                     "Retrieval is done without feedback.")

    if not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
