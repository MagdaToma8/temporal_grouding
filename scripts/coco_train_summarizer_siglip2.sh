python -m src.train_summarizer \
--dataset coco \
--data_config configs/coco/data_summarizer_siglip2.yaml \
--model_family siglip2 \
--model_id google/siglip2-so400m-patch14-384 \
--experiment_config ./configs/coco/siglip2_local_summarizer_nocaploss.yaml \
--num_workers 0