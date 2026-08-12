#!/bin/bash
#SBATCH --job-name=embed_viclip_msrvtt
#SBATCH --output=stdout-%x-%j.out
#SBATCH --error=stderr-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=research
#SBATCH --account=research

cd ~/vlm_retrieval_relevance_feedback
CKPT=$(find checkpoints/backbone -name "epoch=13-val_loss=0.2896.ckpt")
echo "Using checkpoint: $CKPT"

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.run_embeddings_and_retrieval \
    --dataset msrvtt \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --embeddings_dir embeddings/msrvtt/viclip/train \
    --batch_size 16 \
    --split train \
    --save_top_k 10"

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.run_embeddings_and_retrieval \
    --dataset msrvtt \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --embeddings_dir embeddings/msrvtt/viclip/val \
    --batch_size 16 \
    --split val \
    --save_top_k 10"
