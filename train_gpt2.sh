#!/bin/bash
#SBATCH --job-name=train_gpt2
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
# полный прогон ~224 ч, так что дозапуски все равно нужны; чем больше
# лимит, тем их меньше (4 вместо 14)
#SBATCH --time=72:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"
module purge
module load Python
conda deactivate
conda activate pgwtd
export WANDB_MODE=offline
ARCH_TYPE="${ARCH_TYPE:-gpt}"
# число процессов torchrun = число GPU; для замерочных прогонов NPROC=1
NPROC="${NPROC:-4}"
source run_flags.sh

echo "Starting GPT2 training..."
# порт из номера задания: иначе коллизия с train_diffusion на общей ноде (оба брали бы 29500)
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=${NPROC} train_gpt2.py --project_name='actdm' ${ARCH_FLAGS} ${DATA_FLAGS}
echo "GPT2 training finished."