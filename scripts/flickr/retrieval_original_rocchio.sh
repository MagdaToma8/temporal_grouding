MODEL_FAMILY="blip2-embeddings"
MODEL_ID="Salesforce/blip2-itm-vit-g"

for feedback_aggregation in None images generated_captions gt_user; do
    python -m src.retrieval_pipeline \
        --dataset flickr \
        --data_config ./configs/flickr30k/data.yaml \
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