MODEL_FAMILY="blip2-embeddings"
MODEL_ID="Salesforce/blip2-itm-vit-g"
temperatures=(0.05 0.1 0.5 1 5)

for temp in "${temperatures[@]}"; do
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
        --feedback_aggregation generated_captions \
        --temperature $temp;
done