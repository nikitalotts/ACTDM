#!/bin/bash
#SBATCH --job-name=probe_signal
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Оценка: ~5 мин счета (8 батчей x 21 точка сетки, по 3 прохода классификатора
# и один денойзера на точку) плюс до ~30 мин на загрузку датасета, чекпоинтов и
# прогрев кэшей. Итого ~40 мин; лимит с троекратным запасом.
#
# Запас намеренный: SLURM считает ФАКТИЧЕСКИ потраченное время, а не лимит, так
# что щедрый лимит не стоит ни одного GPU-часа -- он только чуть ухудшает
# позицию в очереди. А срез по лимиту стоил бы всего прогона.
# Переопределяется: TIME_LIMIT=... ./run_wikipedia.sh probes-window
#SBATCH --time=02:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Probe: время жизни условного сигнала (accuracy по уровню шума)"

torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 probe_signal_survival.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Probe finished."
