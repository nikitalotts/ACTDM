#!/bin/bash
#SBATCH --job-name=eval_gpt2
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"
module purge
module load Python
conda deactivate
conda activate pgwtd

export WANDB_MODE=offline

ARCH_TYPE="${ARCH_TYPE:-gpt}"
source run_flags.sh

DECODING="greedy"

TEMPERATURE=1.0
TOP_P=0.95
TOP_K=0

echo "Starting GPT-2 evaluation (decoding=${DECODING})..."

# порт из номера задания: на общей ноде дефолтный 29500 может быть занят соседним заданием
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 eval_gpt2.py \
    --project_name actdm \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --decoding ${DECODING} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K}

echo "GPT-2 evaluation finished."