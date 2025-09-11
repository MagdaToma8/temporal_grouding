DATASET=coco #flickr
DATASET_FULL=coco #flickr30k

python -m src.train_summarizer \
--dataset $DATASET \
--data_config configs/$DATASET_FULL/data_summarizer_clip_large.yaml \
--model_family clip \
--model_id openai/clip-vit-large-patch14 \
--experiment_config configs/$DATASET/clip_large_local_summarizer_nocaploss_large.yaml \
--num_workers 2