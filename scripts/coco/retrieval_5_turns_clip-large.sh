MODEL_FAMILY="clip"
MODEL_ID="openai/clip-vit-large-patch14"

# for feedback_aggregation in images generated_captions gt_user; do
#     python -m src.retrieval_pipeline \
#         --dataset coco \
#         --data_config ./configs/coco/data.yaml \
#         --model_family $MODEL_FAMILY \
#         --model_id $MODEL_ID  \
#         --batch_size 2 \
#         --text_from caption_0 \
#         --no_plots \
#         --num_turns 5 \
#         --top_k_feedback 5 \
#         --feedback_aggregation $feedback_aggregation \
#         --temperature 0.05 \
#         --wandb_log_all_turns \
#         --accumulate_feedback;
# done 

python -m src.retrieval_pipeline \
    --dataset coco \
    --data_config ./configs/coco/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 5 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/coco/clip_large_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-16_19_14_33_490377/epoch=19-val_loss=0.33.ckpt \
    --wandb_log_all_turns \
    --accumulate_feedback;