MODEL_FAMILY="clip"
MODEL_ID="openai/clip-vit-base-patch32"
# for top_k_feedback in {1..10}; do
#     for feedback_aggregation in images generated_captions; do
#         python -m src.retrieval_pipeline \
#             --dataset flickr \
#             --data_config ./configs/flickr30k/data.yaml \
#             --model_family $MODEL_FAMILY \
#             --model_id $MODEL_ID  \
#             --batch_size 2 \
#             --text_from caption_0 \
#             --no_plots \
#             --num_turns 2 \
#             --top_k_feedback $top_k_feedback \
#             --feedback_aggregation $feedback_aggregation \
#             --temperature 0.05;
#     done
# done

for top_k_feedback in {1..10}; do
    python -m src.retrieval_pipeline \
        --dataset flickr \
        --data_config ./configs/flickr30k/data.yaml \
        --model_family $MODEL_FAMILY \
        --model_id $MODEL_ID \
        --batch_size 2 \
        --text_from caption_0 \
        --no_plots \
        --num_turns 2 \
        --top_k_feedback $top_k_feedback \
        --feedback_aggregation attentive_summarizer \
        --experiment_config configs/flickr30k/clip_local_summarizer_nocaploss.yaml \
        --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-19_18_49_57_013812/epoch\=25-val_loss\=0.08.ckpt 
done