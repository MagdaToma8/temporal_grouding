#!/bin/bash
#SBATCH --job-name=train_afs_viclip
#SBATCH --output=stdout-%x-%j.out
#SBATCH --error=stderr-%x-%j.err
#SBATCH --time=24:00:00
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

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.train_summarizer \
    --experiment_config configs/msrvtt/viclip_global_summarizer.yaml \
    --data_config configs/msrvtt/data_summarizer_viclip.yaml \
    --dataset msrvtt \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --num_workers 8 \
    --disable_wandb"
