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
    --experiment_config configs/flickr30k/clip_local_summarizer_nocaploss.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-base-patch32-2025-03-19_18_49_57_013812/epoch=25-val_loss=0.08.ckpt \
    --summarizer_no_captions;

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
    --experiment_config configs/flickr30k/clip_large_local_summarizer_nocaploss.yaml \
    --summarizer_checkpoint checkpoints/clip-vit-large-patch14-2025-03-24_14_56_07_315049/epoch=21-val_loss=0.08.ckpt \
    --summarizer_no_captions;

python -m src.retrieval_pipeline \
    --dataset flickr \
    --data_config ./configs/flickr30k/data.yaml \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 2 \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --experiment_config configs/flickr30k/blip2_local_summarizer_nocaploss.yaml \
    --summarizer_checkpoint checkpoints/blip2-itm-vit-g-2025-03-28_14_13_14_067447/epoch=20-val_loss=0.09.ckpt \
    --summarizer_no_captions