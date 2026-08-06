#!/bin/bash
#SBATCH --job-name=finetune_viclip
#SBATCH --output=stdout-%x-%j.out
#SBATCH --error=stderr-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=research
#SBATCH --account=research

cd ~/vlm_retrieval_relevance_feedback
srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.train_backbone \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --batch_size 16 \
    --num_workers 8 \
    --max_epochs 15 \
    --disable_wandb"
