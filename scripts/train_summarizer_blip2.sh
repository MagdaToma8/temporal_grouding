python -m src.train_summarizer \
--data_config configs/flickr30k/data_summarizer_blip2.yaml \
--model_family blip2-embeddings \
--model_id Salesforce/blip2-itm-vit-g \
--experiment_config ./configs/flickr30k/blip2_local_summarizer.yaml \
--num_workers 8