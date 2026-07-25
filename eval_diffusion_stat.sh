#!/bin/bash
#SBATCH --job-name=eval_tencm_diffusion_stat
#SBATCH --output=%j-eval_tencm_diffusion_stat.log
#SBATCH --error=%j-eval_tencm_diffusion_stat.log
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

source run_flags.sh

BASE_SEED=0
NUM_SEEDS=20
SEED_STEP=1000

echo "Starting DIFFUSION STATISTICAL evaluation"
echo "  base_seed=${BASE_SEED}, num_seeds=${NUM_SEEDS}, seed_step=${SEED_STEP}"

torchrun --master_port=31252 --nproc_per_node=1 eval_diffusion_stat.py \
    --dataset_name rocstories \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed ${BASE_SEED} \
    --num_seeds ${NUM_SEEDS} \
    --seed_step ${SEED_STEP}

echo "Diffusion statistical evaluation finished."
