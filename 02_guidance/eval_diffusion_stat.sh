#!/bin/bash
#SBATCH --job-name=eval_cg_diffusion_stat_cg
#SBATCH --output=%j-eval_cg_diffusion_stat_cg.log
#SBATCH --error=%j-eval_cg_diffusion_stat_cg.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=20:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate
conda activate pgwtd

export WANDB_MODE=offline

BASE_SEED=0
NUM_SEEDS=20
SEED_STEP=1000

CG_SCALE=10.0

echo "Starting DIFFUSION STATISTICAL evaluation (classifier guidance)"
echo "  base_seed=${BASE_SEED}, num_seeds=${NUM_SEEDS}, seed_step=${SEED_STEP}"
echo "  classifier_guidance_scale=${CG_SCALE}"

torchrun --master_port=31250 --nproc_per_node=1 eval_diffusion_stat.py \
    --dataset_name rocstories \
    --scheduler sd \
    --coef_d 9 \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --mode transformer \
    --project_name='pgwtd' \
    --eval \
    --use_conditional_encoder \
    --is_conditional \
    --classifier_guidance_scale=${CG_SCALE} \
    --seed ${BASE_SEED} \
    --num_seeds ${NUM_SEEDS} \
    --seed_step ${SEED_STEP}

echo "Diffusion statistical evaluation (classifier guidance) finished."
