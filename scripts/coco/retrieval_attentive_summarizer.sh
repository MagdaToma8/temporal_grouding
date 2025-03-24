# python -m src.retrieval_pipeline \
#     --dataset coco \
#     --data_config ./configs/coco/data.yaml \
#     --model_family clip \
#     --model_id openai/clip-vit-base-patch32  \
#     --batch_size 2 \
#     --text_from caption_0 \
#     --no_plots \
#     --num_turns 2 \
#     --top_k_feedback 5 \
#     --feedback_aggregation attentive_summarizer \
#     --experiment_config configs/coco/clip_local_summarizer.yaml \
#     --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-14_00_43_58_714608/epoch=27-val_loss=0.29.ckpt

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
#     --experiment_config configs/coco/clip_large_local_summarizer.yaml \
#     --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-16_19_14_33_490377/epoch=19-val_loss=0.33.ckpt

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
#     --experiment_config configs/coco/blip2_local_summarizer.yaml \
#     --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-18_18_16_08_180918/epoch=28-val_loss=0.25.ckpt


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
    --experiment_config configs/coco/clip_local_summarizer_noimgloss.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-20_09_53_04_942842/epoch=22-val_loss=0.07.ckpt;


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
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-20_18_19_16_763459/epoch=24-val_loss=0.08.ckpt;