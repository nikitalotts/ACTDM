#!/bin/bash
#SBATCH --job-name=eval_diffusion_stat
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=40:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

# BASE_SEED задаётся снаружи, чтобы прогон на 20 сидов можно было разбить на
# два задания по 10 (BASE_SEED=0 и BASE_SEED=10000 при SEED_STEP=1000 дают
# РОВНО те же сиды 0..19000, что одно задание на 20). Имя дампа несёт
# base_seed, поэтому задания не перезаписывают друг друга, а merge_stat_eval.py
# склеивает их по сырым значениям без потери точности.
BASE_SEED="${BASE_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-20}"
SEED_STEP="${SEED_STEP:-1000}"

echo "Starting DIFFUSION STATISTICAL evaluation"
echo "  base_seed=${BASE_SEED}, num_seeds=${NUM_SEEDS}, seed_step=${SEED_STEP}"

# порт из номера задания: несколько ARCH_TYPE могут попасть на одну ноду
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 eval_diffusion_stat.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed ${BASE_SEED} \
    --num_seeds ${NUM_SEEDS} \
    --seed_step ${SEED_STEP}

echo "Diffusion statistical evaluation finished."
