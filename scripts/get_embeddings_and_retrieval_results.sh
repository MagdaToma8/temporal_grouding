python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-base-patch32/test \
    --split test;

python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-base-patch32/val \
    --split val;

python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 32 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-base-patch32/train \
    --split train;