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
    --rocchio_alpha 0.33 \
    --rocchio_beta 0.33 \
    --rocchio_gamma 0.33;


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
    --rocchio_alpha 0.6 \
    --rocchio_beta 0.2 \
    --rocchio_gamma 0.2;


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
    --rocchio_alpha 0.8 \
    --rocchio_beta 0.2 \
    --rocchio_gamma 0;


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
    --rocchio_alpha 0.8 \
    --rocchio_beta 0 \
    --rocchio_gamma 0.2;


# GRF

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
    --feedback_aggregation generated_captions \
    --rocchio_alpha 0.33 \
    --rocchio_beta 0.33 \
    --rocchio_gamma 0.33;

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
    --feedback_aggregation generated_captions \
    --rocchio_alpha 0.6 \
    --rocchio_beta 0.2 \
    --rocchio_gamma 0.2;

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
    --feedback_aggregation generated_captions \
    --rocchio_alpha 0.8 \
    --rocchio_beta 0.2 \
    --rocchio_gamma 0;

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
    --feedback_aggregation generated_captions \
    --rocchio_alpha 0.8 \
    --rocchio_beta 0 \
    --rocchio_gamma 0.2;