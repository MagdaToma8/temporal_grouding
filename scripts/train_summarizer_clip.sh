DATASET=coco
DATASET_FULL=coco

python -m src.train_summarizer \
--dataset $DATASET \
--data_config configs/$DATASET_FULL/data_summarizer_clip.yaml \
--model_family clip \
--model_id openai/clip-vit-base-patch32 \
--experiment_config configs/$DATASET_FULL/clip_local_summarizer_nocaploss.yaml \
--num_workers 4