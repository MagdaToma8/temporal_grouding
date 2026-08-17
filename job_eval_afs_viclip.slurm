#!/bin/bash
#SBATCH --job-name=eval_afs_viclip
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
BACKBONE_CKPT=$(find checkpoints/backbone -name "epoch=13-val_loss=0.2896.ckpt")
SUMMARIZER_CKPT="checkpoints/ViCLIP-B-16-hf-2026-08-13_09_59_06_085899/epoch=23-val_loss=0.44.ckpt"
echo "Backbone checkpoint: $BACKBONE_CKPT"
echo "Summarizer checkpoint: $SUMMARIZER_CKPT"

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.retrieval_pipeline \
    --dataset msrvtt \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $BACKBONE_CKPT \
    --experiment_config configs/msrvtt/viclip_global_summarizer.yaml \
    --summarizer_checkpoint $SUMMARIZER_CKPT \
    --batch_size 8 \
    --split test \
    --text_from caption_0 \
    --no_plots \
    --num_turns 2 \
    --top_k_feedback 5 \
    --feedback_aggregation attentive_summarizer \
    --summarizer_no_captions \
    --disable_wandb"
