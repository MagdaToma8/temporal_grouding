# DATASET="flickr"
# DATA_CONFIG="./configs/flickr30k/data.yaml"

DATASET="coco"
DATA_CONFIG="./configs/coco/data.yaml"

# EMBEDDINGS_DIR="embeddings/flickr30k/blip2-itm-vit-g/test"
# MODEL_FAMILY="blip2-embeddings"
# MODEL_ID="Salesforce/blip2-itm-vit-g"

EMBEDDINGS_DIR="embeddings/coco/clip-vit-large-patch14"
MODEL_FAMILY="clip"
MODEL_ID="openai/clip-vit-large-patch14"

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 5 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/test \
    --split test;

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 5 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/val \
    --split val;

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 5 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/train \
    --split train;