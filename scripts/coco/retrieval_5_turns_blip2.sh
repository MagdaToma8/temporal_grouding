MODEL_FAMILY="blip2-embeddings"
MODEL_ID="Salesforce/blip2-itm-vit-g"

for feedback_aggregation in images generated_captions gt_user; do
    python -m src.retrieval_pipeline \
        --dataset coco \
        --data_config ./configs/coco/data.yaml \
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
    --dataset coco \
    --data_config ./configs/coco/data.yaml \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 5 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/coco/blip2_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-18_18_16_08_180918/epoch=28-val_loss=0.25.ckpt \
    --wandb_log_all_turns \
    --accumulate_feedback;