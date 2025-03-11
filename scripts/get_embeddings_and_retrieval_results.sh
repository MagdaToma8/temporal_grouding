python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 16 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-large-patch14/test \
    --split test;

python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 16 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-large-patch14/val \
    --split val;

python -m src.run_embeddings_and_retrieval \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 16 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/clip-vit-large-patch14/train \
    --split train;