#!/bin/bash
#SBATCH --job-name=train_gpt2
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=16:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"
module purge
module load Python
conda deactivate
conda activate pgwtd
export WANDB_MODE=offline
ARCH_TYPE="${ARCH_TYPE:-gpt}"
source run_flags.sh

echo "Starting GPT2 training..."
# порт из номера задания: иначе коллизия с train_diffusion на общей ноде (оба брали бы 29500)
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=4 train_gpt2.py --project_name='actdm' ${ARCH_FLAGS} ${DATA_FLAGS}
echo "GPT2 training finished."