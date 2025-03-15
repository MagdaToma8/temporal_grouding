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
    --experiment_config configs/coco/clip_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-14_00_43_58_714608/epoch=27-val_loss=0.29.ckpt

# python -m src.retrieval_pipeline \
#     --dataset flickr \
#     --data_config ./configs/flickr30k/data.yaml \
#     --model_family clip \
#     --model_id openai/clip-vit-large-patch14  \
#     --batch_size 2 \
#     --text_from caption_0 \
#     --no_plots \
#     --num_turns 2 \
#     --top_k_feedback 5 \
#     --feedback_aggregation attentive_summarizer \
#     --experiment_config configs/flickr30k/clip_large_local_summarizer.yaml \
#     --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-04_23_38_07_224736/epoch\=18-val_loss\=0.35.ckpt

# python -m src.retrieval_pipeline \
#     --dataset flickr \
#     --data_config ./configs/flickr30k/data.yaml \
#     --model_family blip2-embeddings \
#     --model_id Salesforce/blip2-itm-vit-g \
#     --batch_size 2 \
#     --text_from caption_0 \
#     --no_plots \
#     --num_turns 2 \
#     --top_k_feedback 5 \
#     --feedback_aggregation attentive_summarizer \
#     --experiment_config configs/flickr30k/blip2_local_summarizer.yaml \
#     --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-12_12_22_44_731160/epoch=29-val_loss=0.25.ckpt
