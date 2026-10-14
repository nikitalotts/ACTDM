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
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

# $0 под sbatch -- это копия скрипта в /var/spool/slurm, а не файл в репозитории,
# поэтому dirname "$0" ведет мимо hf_env.sh. Рабочий каталог задания --
# SLURM_SUBMIT_DIR; вне кластера его нет, и остается обычный dirname.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/hf_env.sh"
export PYTHONUNBUFFERED=1

# смотрим ровно ту карту, что дал slurm
python find_max_batch.py --dataset "${DATASET:-wikipedia}" \
    --arch ${ARCHS:-gpt genie diffuseq unconditional} \
    ${MAX_BATCH:+--max-batch ${MAX_BATCH}}
