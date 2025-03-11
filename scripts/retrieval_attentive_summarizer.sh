python -m src.retrieval_pipeline \
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/flickr30k/clip_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-02-27_00_44_29_864193/epoch\=28-val_loss\=0.31.ckpt 

python -m src.retrieval_pipeline \
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14  \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/flickr30k/clip_large_local_summarizer.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-04_23_38_07_224736/epoch\=18-val_loss\=0.35.ckpt 