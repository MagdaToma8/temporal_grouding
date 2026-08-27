import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.datasets.flickr import load_flickr_data
from src.datasets.coco import load_coco_data
from src.datasets.msrvtt import load_msrvtt_data
from src.models.attentive_summarizer import AttentiveSummarizer, AlignmentAttentiveSummarizer
from src.models.clip_video_finetuner import load_finetuned_clip_state_dict
from src.models.configs import get_model_config
from src.models.viclip import VICLIP_CONTEXT_LENGTH
from src.utils.utils import load_yaml_file, generate_experiment_id
from src.utils.quantization import bitsandbytes_8bit_config


def parse_args():
    parser = ArgumentParser()
    # config files
    parser.add_argument('--experiment_config', type=str, required=True)
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='flickr')
    # VLM model
    parser.add_argument('--model_family', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument(
        '--backbone_checkpoint', type=str, default=None,
        help="Path to a train_backbone.py checkpoint -- AttentiveSummarizer re-encodes "
             "retrieved items' raw pixels/text at training time (not just the pre-computed "
             "embeddings run_embeddings_and_retrieval.py used to pick them), so this should "
             "match whatever backbone generated --data_config's embeddings_path."
    )
    # optimization args
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    # processing args
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true', default=False)
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help="Run 2 train/val batches only, to check the training loop is wired correctly"
    )
    # outputs args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_id = generate_experiment_id()
    log_id = f"{args.model_id.split('/')[-1]}-{experiment_id}"

    experiment_config = load_yaml_file(args.experiment_config)
    data_config = load_yaml_file(args.data_config)

    model_config = get_model_config(args.model_family, args.model_id)
    base_model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        quantization_config=bitsandbytes_8bit_config() if args.use_8bit else None,
        trust_remote_code=True
    )
    if args.backbone_checkpoint:
        base_model.load_state_dict(load_finetuned_clip_state_dict(args.backbone_checkpoint))
        print(f"Loaded fine-tuned backbone weights from {args.backbone_checkpoint}")
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])

    vlm_wrapper = model_config["wrapper_class"](model=base_model, processor=processor)

    if args.dataset == 'flickr':
        train_dataset, train_collator = load_flickr_data(
            data_config,
            'train',
            processor,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
        val_dataset, val_collator = load_flickr_data(
            data_config,
            'val',
            processor,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
    elif args.dataset == 'coco':
        train_dataset, train_collator = load_coco_data(
            data_config,
            'train',
            processor,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
        val_dataset, val_collator = load_coco_data(
            data_config,
            'val',
            processor,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
    elif args.dataset == 'msrvtt':
        # process_images=True: unlike load_flickr_data/load_coco_data (which default to
        # True), load_msrvtt_data defaults to False -- every other caller in this codebase
        # passes it explicitly, so this matches that convention.
        train_dataset, train_collator = load_msrvtt_data(
            data_config,
            'train',
            processor,
            process_images=True,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
        val_dataset, val_collator = load_msrvtt_data(
            data_config,
            'val',
            processor,
            process_images=True,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or experiment_config.get("batch_size", 128),
        num_workers=args.num_workers,
        collate_fn=train_collator,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or experiment_config.get("batch_size", 128),
        num_workers=args.num_workers,
        collate_fn=val_collator
    )

    # Initialize summarizer model
    summarizer = AttentiveSummarizer(
        pooler_config=experiment_config["pooler_config"],
        text_dim_local=experiment_config.get("text_dim_local", experiment_config.get("text_dim", 768)),
        text_dim_global=experiment_config.get("text_dim_global", experiment_config.get("text_dim", 768)),
        vision_dim=experiment_config["vision_dim"],
        vlm_wrapper=vlm_wrapper,
        global_embeddings_vision=experiment_config.get("global_embeddings_vision", True),
        global_embeddings_text=experiment_config.get("global_embeddings_text", True),
        random_mask=experiment_config.get("random_mask", False),
        video_num_frames=data_config.get("num_frames") if args.dataset == "msrvtt" else None,
        # ViCLIP's text encoder has a fixed 32-token context length (no slicing on shorter
        # input, unlike CLIP), so _get_vision_features' dummy input_ids must match exactly.
        text_seq_len=VICLIP_CONTEXT_LENGTH if args.model_family == "viclip" else 10,
    )

    # Initialize Lightning module
    model = AlignmentAttentiveSummarizer(
        summarizer=summarizer,
        pooler_config=experiment_config["pooler_config"],
        learning_rate=args.learning_rate or experiment_config.get("learning_rate", 1e-4),
        weight_decay=args.weight_decay or experiment_config.get("weight_decay", 0.01),
        max_epochs=args.max_epochs or experiment_config.get("max_epochs", 100),
        random_mask=experiment_config.get("random_mask", False),
        no_image_loss=experiment_config.get("no_image_loss", False),
        no_caption_loss=experiment_config.get("no_caption_loss", False)
    )

    # Setup training
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, log_id),
        filename='epoch={epoch}-val_loss={val/loss:.2f}',
        save_top_k=1,
        monitor='val/loss',
        auto_insert_metric_name=False
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0005,
        patience=10,
        verbose=True
    )

    logger = False
    if not args.disable_wandb:
        logger = WandbLogger(
            entity='sensor_har',
            project=f'{args.dataset}-summarizer-training',
            name=log_id,
            config={
                **data_config,
                **experiment_config
            }
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs or experiment_config.get("max_epochs", 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            checkpoint_callback,
            ModelSummary(max_depth=2),
            early_stopping_callback
        ],
        logger=logger,
        log_every_n_steps=5,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        limit_train_batches=2 if args.debug else 1.0,
        limit_val_batches=2 if args.debug else 1.0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    if torch.cuda.is_available():
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
