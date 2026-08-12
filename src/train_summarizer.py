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
from src.models.configs import get_model_config
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
    # optimization args
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    # processing args
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_8bit', action='store_true')
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
        train_dataset, train_collator = load_msrvtt_data(
            data_config,
            'train',
            processor,
            summarizer=True,
            siglip2=True if "siglip" in args.model_family else False
        )
        val_dataset, val_collator = load_msrvtt_data(
            data_config,
            'val',
            processor,
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
        video_num_frames=data_config.get("num_frames") if args.dataset == "msrvtt" else None
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

    wandb_logger = WandbLogger(
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
        logger=wandb_logger,
        log_every_n_steps=5,
        precision="bf16-mixed"
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    main()
