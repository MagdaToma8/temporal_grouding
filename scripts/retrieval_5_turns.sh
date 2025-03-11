MODEL_FAMILY="clip"
MODEL_ID="openai/clip-vit-base-patch32"

for feedback_aggregation in images generated_captions gt_user; do
    python -m src.retrieval_pipeline \
        --dataset flickr \
        --data_config ./configs/flickr30k/data.yaml \
        --model_family $MODEL_FAMILY \
        --model_id $MODEL_ID  \
        --batch_size 2 \
        --text_from caption_0 \
        --no_plots \
        --num_turns 5 \
        --top_k_feedback 5 \
        --feedback_aggregation $feedback_aggregation \
        --temperature 0.05 \
        --wandb_log_all_turns \
        --accumulate_feedback;
done

python -m src.retrieval_pipeline \
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 5 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/flickr30k/clip_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-02-27_00_44_29_864193/epoch\=28-val_loss\=0.31.ckpt \
    --wandb_log_all_turns \
    --accumulate_feedback;