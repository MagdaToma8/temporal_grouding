from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoProcessor


class ImageTextDataCollator:
    def __init__(self, processor: Optional[AutoProcessor] = None, siglip2=False):
        self.processor = processor
        self.siglip2 = siglip2

    def __call__(self, batch: List[Dict]) -> Dict:
        processed_batch = {}
        for key in batch[0].keys():
            if key != "image" and key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])

        if self.processor is not None:
            if not self.siglip2:
                processed_img_text = self.processor(
                    images=[example['image'] for example in batch],
                    text=[example['text_class_label'] for example in batch],
                    return_tensors="pt",
                    padding=True
                )
            else:
                processed_img_text = self.processor(
                    images=[example['image'] for example in batch],
                    text=[example['text_class_label'].lower() for example in batch],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
            processed_batch['image'] = processed_img_text['pixel_values']
            processed_batch['input_ids'] = processed_img_text['input_ids']
            processed_batch['attention_mask'] = processed_img_text['attention_mask']
        return processed_batch


class CaptioningDataCollator:
    def __init__(
            self,
            processor: AutoProcessor = None,
            process_images: bool = True,
            num_captions: int = 5,
            siglip2=False,
            is_video=False,
        ):
        self.processor = processor
        self.process_images = process_images
        self.num_captions = num_captions
        self.siglip2 = siglip2
        # when True, each example's "image" is a list of num_frames PIL frames instead
        # of a single image (e.g. MSRVTTDataset). Frames are flattened across the whole
        # batch for one processor call, then reshaped back to [batch, num_frames, C, H, W].
        self.is_video = is_video

    def __call__(self, batch):
        processed_batch = {}

        processed_batch['img_path'] = np.array([example['img_path'] for example in batch])
        processed_batch['class_label'] = torch.tensor([example['class_label'] for example in batch])

        if self.process_images and self.processor is not None:
            if self.is_video:
                num_frames = len(batch[0]['image'])
                flattened_frames = [frame for example in batch for frame in example['image']]
                processed_images = self.processor(images=flattened_frames, return_tensors="pt")
                pixel_values = processed_images['pixel_values']  # [batch*num_frames, C, H, W]
                processed_batch['image'] = pixel_values.view(len(batch), num_frames, *pixel_values.shape[1:])
            elif not self.siglip2:
                processed_img_text = self.processor(
                    images=[example['image'] for example in batch],
                    text=[example[f"caption_{0}"] for example in batch],
                    return_tensors="pt",
                    padding=True
                )
                processed_batch['image'] = processed_img_text['pixel_values']
            else:
                processed_img_text = self.processor(
                    images=[example['image'] for example in batch],
                    text=[example[f"caption_{0}"].lower() for example in batch],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
                processed_batch['image'] = processed_img_text['pixel_values']
        else:
            processed_batch['image'] = [example['image'] for example in batch]

        if self.processor is not None:
            for i in range(self.num_captions):
                if not self.siglip2:
                    processed_text = self.processor(
                        text=[example[f"caption_{i}"] for example in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                else:
                    processed_text = self.processor(
                        text=[example[f"caption_{i}"].lower() for example in batch],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                        truncation=True,
                    )
                processed_batch[f"caption_{i}"] = processed_text['input_ids']
                processed_batch[f'caption_{i}_attention_mask'] = processed_text['attention_mask'] if f'caption_{i}_attention_mask' in processed_text else None
        return processed_batch


class SummarizerDatasetCollator:
    def __init__(self, processor=None, process_images=True, siglip2=False):
        self.processor = processor
        self.process_images = process_images
        self.siglip2 = siglip2

    def __call__(self, batch):
        processed_batch = {}

        # Process image paths and class labels
        processed_batch['img_path'] = np.array([example['img_path'] for example in batch])
        processed_batch['class_label'] = torch.tensor([example['class_label'] for example in batch])

        # Process ground truth images
        if self.process_images and self.processor is not None:
            if not self.siglip2:
                processed_images = self.processor(
                images=[example['image'] for example in batch],
                    return_tensors="pt",
                    padding=True
                )
            else:
                processed_images = self.processor(
                    images=[example['image'] for example in batch],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
            processed_batch['image'] = processed_images['pixel_values']
        else:
            processed_batch['image'] = [example['image'] for example in batch]

        # Process text fields
        if self.processor is not None:
            # Process query
            if not self.siglip2:
                processed_query = self.processor(
                    text=[example['query'] for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            else:
                processed_query = self.processor(
                    text=[example['query'].lower() for example in batch],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
            processed_batch['query_input_ids'] = processed_query['input_ids']
            processed_batch['query_attention_mask'] = processed_query['attention_mask'] if "attention_mask" in processed_query else None # type: ignore

            # Process ground truth captions
            processed_batch["num_ground_truth_captions"] = len(batch[0]["ground_truth"])
            all_ground_truth_captions = []
            # Flatten ground truth captions for each example along the batch dimension
            # Shape for input_ids and ground_truth_attention_mask:
            #   [bsz * (num_captions - 1), seq_len]
            #   seq_len will be padded to the longest ground truth caption in the batch
            for example in batch:
                all_ground_truth_captions.extend(example["ground_truth"])
            if not self.siglip2:
                processed_ground_truth = self.processor(
                    text=all_ground_truth_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                    )
            else:
                processed_ground_truth = self.processor(
                    text=[caption.lower() for caption in all_ground_truth_captions],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                )
            processed_batch['ground_truth_input_ids'] = processed_ground_truth['input_ids']
            processed_batch['ground_truth_attention_mask'] = processed_ground_truth['attention_mask'] if "attention_mask" in processed_ground_truth else None # type: ignore

            # Process text feedback from yolo-based models
            # Shape for input_ids and attention_mask:
            #   [bsz, seq_len]
            if example.get('text_feedback', None) and any(example['text_feedback'] for example in batch):
                if not self.siglip2:
                    processed_text_feedback = self.processor(
                        text=[' '.join(example['text_feedback']) for example in batch],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                else:
                    processed_text_feedback = self.processor(
                        text=[' '.join(example['text_feedback'].lower()) for example in batch],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                        truncation=True,
                    )
                processed_batch['text_feedback_input_ids'] = processed_text_feedback['input_ids']
                processed_batch['text_feedback_attention_mask'] = processed_text_feedback['attention_mask'] if "attention_mask" in processed_text_feedback else None # type: ignore

            # Process LLaVA-generated captions
            # Flatten generated captions for each retrieved image in topk along the batch dimension
            # Shape for input_ids and attention_mask:
            #   [bsz * topk, seq_len]
            #   seq_len will be padded to the longest generated caption in the batch
            if any(example['generated_text'] for example in batch):
                processed_batch["num_generated_captions"] = len(batch[0]['generated_text'])
                all_generated_captions = []
                for example in batch:
                    all_generated_captions.extend(example['generated_text'])
                if not self.siglip2:
                    processed_generated_text = self.processor(
                        text=all_generated_captions,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                else:
                    processed_generated_text = self.processor(
                        text=[caption.lower() for caption in all_generated_captions],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                        truncation=True,
                    )
                processed_batch['generated_text_input_ids'] = processed_generated_text['input_ids']
                processed_batch['generated_text_attention_mask'] = processed_generated_text['attention_mask'] if "attention_mask" in processed_generated_text else None # type: ignore

        # Process retrieval results images
        # Flatten topk retrieved images along the batch dimension
        # Shape for pixel_values:
        #   [bsz * topk, 3, 224, 224]
        if self.process_images and self.processor is not None:
            retrieved_images = [img for example in batch for img in example['retrieval_results_images']]
            processed_retrieval_images = self.processor(
                images=retrieved_images,
                return_tensors="pt",
                padding=True
            )
            processed_batch['retrieval_results_images'] = processed_retrieval_images['pixel_values']
        else:
            processed_batch['retrieval_results_images'] = [example['retrieval_results_images'] for example in batch]

        return processed_batch
