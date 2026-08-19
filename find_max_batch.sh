#!/bin/bash
#SBATCH --job-name=find_max_batch
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

# Подбор максимального батча на одну карту для каждого подхода.
# Датасет не читается, поэтому задание короткое: минуты.
#
#   sbatch find_max_batch.sh                      все подходы
#   ARCHS="gpt" sbatch find_max_batch.sh          только gpt
#
# Результат -- таблица в slurm_logs/<jobid>-find_max_batch.log

source ~/.bashrc
eval "$(conda shell.bash hook)" 2>/dev/null || true
module purge
module load Python
conda deactivate
conda activate pgwtd

source "$(dirname "$0")/hf_env.sh"
export PYTHONUNBUFFERED=1

# смотрим ровно ту карту, что дал slurm
python find_max_batch.py --dataset "${DATASET:-wikipedia}" \
    --arch ${ARCHS:-gpt genie diffuseq unconditional} \
    ${MAX_BATCH:+--max-batch ${MAX_BATCH}}
