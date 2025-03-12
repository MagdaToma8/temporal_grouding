python -m src.run_embeddings_and_retrieval \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 5 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/blip2-itm-vit-g/test \
    --split test;

python -m src.run_embeddings_and_retrieval \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 5 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/blip2-itm-vit-g/val \
    --split val;

python -m src.run_embeddings_and_retrieval \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 5 \
    --dataset flickr \
    --data_config configs/flickr30k/data.yaml \
    --embeddings_dir embeddings/flickr30k/blip2-itm-vit-g/train \
    --split train;