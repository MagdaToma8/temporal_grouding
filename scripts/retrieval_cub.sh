# !/bin/bash
DATA_CONFIG=$1

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 8 \
    --text_from text_caption \
    --save_embeddings ./embeddings/clip_test_captions.pt;

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family clip \
    --model_id openai/clip-vit-base-patch32 \
    --batch_size 8 \
    --text_from text_caption_and_attribute \
    --save_embeddings ./embeddings/clip_test_captions_and_attributes.pt;

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 8 \
    --text_from text_caption \
    --save_embeddings ./embeddings/clip-vit-large-patch14_test_captions.pt;

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family clip \
    --model_id openai/clip-vit-large-patch14 \
    --batch_size 8 \
    --text_from text_caption_and_attribute \
    --save_embeddings ./embeddings/clip-vit-large-patch14_test_captions_and_attributes.pt;

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 8 \
    --text_from text_caption \
    --save_embeddings ./embeddings/blip2-itm-vit-g_test_captions.pt;

python -m src.retrieval_pipeline \
    --dataset cub \
    --data_config $DATA_CONFIG \
    --model_family blip2-embeddings \
    --model_id Salesforce/blip2-itm-vit-g \
    --batch_size 8 \
    --text_from text_caption_and_attribute \
    --save_embeddings ./embeddings/blip2-itm-vit-g_test_captions_and_attributes.pt;