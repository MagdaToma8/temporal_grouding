MODEL_FAMILY="clip"
MODEL_ID="openai/clip-vit-base-patch32"

for feedback_aggregation in None images generated_captions gt_user; do
    python -m src.retrieval_pipeline \
        --dataset coco \
        --data_config ./configs/coco/data.yaml \
        --model_family $MODEL_FAMILY \
        --model_id $MODEL_ID  \
        --batch_size 2 \
        --text_from caption_0 \
        --no_plots \
        --num_turns 2 \
        --top_k_feedback 5 \
        --feedback_aggregation $feedback_aggregation \
        --temperature 0.05 \
        --original_rocchio;
done