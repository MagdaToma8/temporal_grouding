#!/bin/bash
#SBATCH --job-name=eval_viclip
#SBATCH --output=stdout-%x-%j.out
#SBATCH --error=stderr-%x-%j.err
#SBATCH --time=00:30:00
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

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.analyze_temporal_split \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --labels_file analysis/msrvtt_test_temporal_labels.json \
    --label viclip_finetuned_8frame"

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.analyze_temporal_split \
    --data_config configs/msrvtt/data_viclip_1frame.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --labels_file analysis/msrvtt_test_temporal_labels.json \
    --label viclip_finetuned_1frame"
