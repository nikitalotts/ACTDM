#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
# diffuseq по замеру нужно ~68 ч + 2 ч на eval; genie и unconditional
# укладываются с большим запасом (52 и 37 ч)
#SBATCH --time=72:00:00


source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate 
conda activate pgwtd

export WANDB_MODE=offline

# число процессов torchrun = число GPU; для замерочных прогонов NPROC=1
NPROC="${NPROC:-4}"

source run_flags.sh

echo "Starting script..."

# порт из номера задания: genie/diffuseq/uncond уходят в очередь одновременно,
# и на общей ноде дефолтный 29500 у второго torchrun занят
torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=${NPROC} train_diffusion.py --encoder_name bert-base-cased --project_name='actdm' ${ARCH_FLAGS} ${DATA_FLAGS}

echo "Script finished."