import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from src.datasets.msrvtt import load_msrvtt_data
from src.models.clip_video_finetuner import CLIPVideoFineTuner
from src.models.configs import get_model_config
from src.utils.utils import load_yaml_file, generate_experiment_id


def parse_args():
    parser = ArgumentParser(description="Contrastive fine-tuning of the video retrieval backbone")
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="clip_video")
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Deliberately small: continuing to train an already-pretrained model, not training from scratch")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--freeze_backbone", action="store_true", default=False,
        help="Partial fine-tuning: freeze the vision/text towers, only train the projection layers"
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/backbone")
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Run 2 train/val batches only, to check the training loop is wired correctly"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_id = generate_experiment_id()
    log_id = f"{args.model_family}-finetune-{experiment_id}"

    data_config = load_yaml_file(args.data_config)

    model_config = get_model_config(args.model_family, args.model_id)
    base_model = model_config["model_class"].from_pretrained(model_config["model_id"])
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])
    vlm_wrapper = model_config["wrapper_class"](model=base_model, processor=processor)

    # Contrastive training wants one (video, caption) pair per example, with a different
    # randomly-sampled caption each epoch -- MSRVTTDataset already does this for free when
    # asked for fewer captions than are available, so just override num_captions_to_use here
    # rather than requiring separate train/val config files.
    train_config = {**data_config, "num_captions_to_use": 1}
    val_config = {**data_config, "num_captions_to_use": 1}

    train_dataset, train_collator = load_msrvtt_data(train_config, "train", processor, process_images=True)
    val_dataset, val_collator = load_msrvtt_data(val_config, "val", processor, process_images=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=val_collator
    )

    model = CLIPVideoFineTuner(
        vlm_wrapper=vlm_wrapper,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        freeze_backbone=args.freeze_backbone,
        temperature=args.temperature,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, log_id),
        filename="epoch={epoch}-val_loss={val/loss:.4f}",
        save_top_k=1,
        monitor="val/loss",
        auto_insert_metric_name=False
    )
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.0005,
        patience=5,
        verbose=True
    )

    logger = False
    if not args.disable_wandb:
        logger = WandbLogger(
            project="msrvtt-backbone-finetune",
            name=log_id,
            config={**data_config, **vars(args)}
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2), early_stopping_callback],
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


if __name__ == "__main__":
    main()
