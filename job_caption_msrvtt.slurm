#!/bin/bash
#SBATCH --job-name=caption_msrvtt
#SBATCH --output=stdout-%x-%j.out
#SBATCH --error=stderr-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=research
#SBATCH --account=research

cd ~/vlm_retrieval_relevance_feedback

# test split first: this is what retrieval_pipeline.py's --feedback_aggregation
# generated_captions (GRF) needs (captions_test.json), the same split used for
# the no-feedback/PRF/AFS comparison in the README.
srun apptainer exec --nv pytorch.sif bash -c "source venv/bin/activate && python -m src.captioning_pipeline \
    --dataset msrvtt \
    --data_config configs/msrvtt/data_viclip.yaml \
    --model_family llava_next_video \
    --model_id llava-hf/LLaVA-NeXT-Video-7B-hf \
    --batch_size 4 \
    --split test \
    --use_8bit \
    --by_image_path"
