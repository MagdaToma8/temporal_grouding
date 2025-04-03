MODEL_FAMILY="blip2-embeddings"
MODEL_ID="Salesforce/blip2-itm-vit-g"

# for feedback_aggregation in images generated_captions gt_user; do
#     python -m src.retrieval_pipeline \
#         --dataset flickr \
#         --data_config ./configs/flickr30k/data.yaml \
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
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 5 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/flickr30k/blip2_local_summarizer_nocaploss.yaml \
    --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-28_14_13_14_067447/epoch=20-val_loss=0.09.ckpt \
    --wandb_log_all_turns \
    --accumulate_feedback;