DATASET=coco

python -m src.train_summarizer \
--dataset $DATASET \
--data_config configs/$DATASET/data_summarizer_clip.yaml \
--model_family clip \
--model_id openai/clip-vit-base-patch32 \
--experiment_config configs/$DATASET/clip_local_summarizer.yaml \
--num_workers 2