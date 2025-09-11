python -m src.train_summarizer \
--dataset flickr \
--data_config configs/flickr30k/data_summarizer_siglip2.yaml \
--model_family siglip2 \
--model_id google/siglip2-so400m-patch14-384 \
--experiment_config ./configs/flickr30k/siglip2_local_summarizer_nocaploss.yaml \
--num_workers 0