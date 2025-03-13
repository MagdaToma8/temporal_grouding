DATASET="coco"
DATA_CONFIG="./configs/coco/data.yaml"

python -m src.captioning_pipeline \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --model_family llava \
    --model_id llava-hf/llava-1.5-7b-hf \
    --batch_size 8 \
    --split test \
    --use_8bit \
    --by_image_path

python -m src.captioning_pipeline \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --model_family llava \
    --model_id llava-hf/llava-1.5-7b-hf \
    --batch_size 8 \
    --split val \
    --use_8bit \
    --by_image_path

python -m src.captioning_pipeline \
    --dataset $DATASET \
    --data_config $DATA_CONFIG \
    --model_family llava \
    --model_id llava-hf/llava-1.5-7b-hf \
    --batch_size 8 \
    --split train \
    --use_8bit \
    --by_image_path