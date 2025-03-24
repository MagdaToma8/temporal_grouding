DATASET=flickr

python -m src.train_summarizer \
--dataset flickr \
--data_config configs/flickr30k/data_summarizer_clip_large.yaml \
--model_family clip \
--model_id openai/clip-vit-large-patch14 \
--experiment_config configs/flickr30k/clip_large_local_summarizer_nocaploss.yaml \
--num_workers 2