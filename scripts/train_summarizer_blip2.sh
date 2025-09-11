python -m src.train_summarizer \
--dataset coco \
--data_config configs/coco/data_summarizer_blip2.yaml \
--model_family blip2-embeddings \
--model_id Salesforce/blip2-itm-vit-g \
--experiment_config ./configs/coco/blip2_local_summarizer_nocaploss_large.yaml \
--num_workers 0