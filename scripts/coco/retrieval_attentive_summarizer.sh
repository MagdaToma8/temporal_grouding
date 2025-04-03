python -m src.retrieval_pipeline \
    --dataset coco \
    --data_config ./configs/coco/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/coco/clip_local_summarizer_nocaploss.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-20_18_19_16_763459/epoch=24-val_loss=0.08.ckpt

# python -m src.retrieval_pipeline \
#     --dataset coco \
#     --data_config ./configs/coco/data.yaml \
#     --model_family clip \
#     --model_id openai/clip-vit-large-patch14  \
#     --batch_size 2 \
#     --text_from caption_0 \
#     --no_plots \
#     --num_turns 2 \
#     --top_k_feedback 5 \
#     --feedback_aggregation attentive_summarizer \
#     --experiment_config configs/coco/clip_large_local_summarizer_nocaploss.yaml \
#     --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-24_15_09_55_874696/epoch=19-val_loss=0.08.ckpt;

# python -m src.retrieval_pipeline \
#     --dataset coco \
#     --data_config ./configs/coco/data.yaml \
#     --model_family blip2-embeddings \
#     --model_id Salesforce/blip2-itm-vit-g \
#     --batch_size 2 \
#     --text_from caption_0 \
#     --no_plots \
#     --num_turns 2 \
#     --top_k_feedback 5 \
#     --feedback_aggregation attentive_summarizer \
#     --experiment_config configs/coco/blip2_local_summarizer_nocaploss.yaml \
#     --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-28_20_14_46_656914/epoch=27-val_loss=0.08.ckpt
