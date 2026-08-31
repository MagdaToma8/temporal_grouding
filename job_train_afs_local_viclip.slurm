#!/bin/bash
#SBATCH --job-name=train_afs_local_viclip
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

# Fixed run name so every submission of this script writes to (and can resume from) the
# same checkpoints/ViCLIP-B-16-hf-afs_local_viclip/ directory -- local-mode AFS training
# doesn't fit inside one 24h job, so this script is meant to be resubmitted with `sbatch`
# repeatedly (once the previous submission's time limit is hit) until early stopping
# triggers or max_epochs (30) is reached. Each resubmission auto-detects and resumes from
# the last checkpoint (epoch, optimizer/scheduler state, and early-stopping counter all
# restored), so no manual bookkeeping is needed between submissions.
RUN_NAME=afs_local_viclip
RESUME_CKPT=$(find checkpoints/ViCLIP-B-16-hf-$RUN_NAME -name "last.ckpt" 2>/dev/null)
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARG="--resume_checkpoint $RESUME_CKPT"
else
    echo "No existing checkpoint found -- starting fresh"
    RESUME_ARG=""
fi

srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.train_summarizer \
    --experiment_config configs/msrvtt/viclip_local_summarizer.yaml \
    --data_config configs/msrvtt/data_summarizer_viclip.yaml \
    --dataset msrvtt \
    --model_family viclip \
    --model_id OpenGVLab/ViCLIP-B-16-hf \
    --backbone_checkpoint $CKPT \
    --run_name $RUN_NAME \
    $RESUME_ARG \
    --num_workers 8 \
    --disable_wandb"
