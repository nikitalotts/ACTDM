#!/bin/bash
#SBATCH --job-name=eval_gpt2_stat
#SBATCH --output=%j-eval_gpt2_stat.log
#SBATCH --error=%j-eval_gpt2_stat.log
#SBATCH --cpus-per-task=4
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

DECODING="sampling"
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=0

BASE_SEED=0
NUM_SEEDS=20
SEED_STEP=1000

echo "Starting GPT-2 STATISTICAL evaluation"
echo "  decoding=${DECODING}"
echo "  base_seed=${BASE_SEED}, num_seeds=${NUM_SEEDS}, seed_step=${SEED_STEP}"

torchrun --master_port=31251 --nproc_per_node=1 eval_gpt2_stat.py \
    --dataset_name rocstories \
    --project_name actdm \
    --decoding ${DECODING} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --seed ${BASE_SEED} \
    --num_seeds ${NUM_SEEDS} \
    --seed_step ${SEED_STEP}

echo "GPT-2 statistical evaluation finished."
