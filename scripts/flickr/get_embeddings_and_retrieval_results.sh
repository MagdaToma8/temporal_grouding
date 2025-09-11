DATASET="flickr"
DATA_CONFIG="./configs/flickr30k/data.yaml"
EMBEDDINGS_DIR="embeddings/flickr30k/siglip2-so400m-patch14-384"
MODEL_FAMILY="siglip2"
MODEL_ID="google/siglip2-so400m-patch14-384"

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 2 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/test \
    --split test;

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 2 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/val \
    --split val;

python -m src.run_embeddings_and_retrieval \
    --model_family $MODEL_FAMILY \
    --model_id $MODEL_ID \
    --batch_size 10 \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --embeddings_dir $EMBEDDINGS_DIR/train \
    --split train \
    --chunk_size 1000;
