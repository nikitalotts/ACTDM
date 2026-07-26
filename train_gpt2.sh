#!/bin/bash
#SBATCH --job-name=train_gpt2
#SBATCH --output=%j-train_gpt2.log
#SBATCH --error=%j-train_gpt2.log
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
torchrun --nproc_per_node=4 train_gpt2.py --project_name='actdm' ${ARCH_FLAGS} ${DATA_FLAGS}
echo "GPT2 training finished."