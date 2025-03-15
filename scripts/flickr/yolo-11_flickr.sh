# !/bin/bash

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/object_detection \
--split test \
--batch_size 16 \
--conf_threshold 0.5 \
--model_path yolo11x.pt;

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/object_detection \
--split val \
--batch_size 16 \
--conf_threshold 0.5 \
--model_path yolo11x.pt;

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/object_detection \
--split train \
--batch_size 16 \
--conf_threshold 0.5 \
--model_path yolo11x.pt;

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/classification \
--split test \
--batch_size 16 \
--conf_threshold 0.15 \
--model_path yolo11x-cls.pt;

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/classification \
--split val \
--batch_size 16 \
--conf_threshold 0.15 \
--model_path yolo11x-cls.pt;

python -m src.run_yolo_detection \
--data_config ./configs/flickr30k/data.yaml \
--dataset flickr \
--output_dir data/flickr30k/yolo/classification \
--split train \
--batch_size 16 \
--conf_threshold 0.15 \
--model_path yolo11x-cls.pt;