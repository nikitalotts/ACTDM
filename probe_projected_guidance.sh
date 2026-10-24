#!/bin/bash
#SBATCH --job-name=probe_projguid
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Оценка: 1000 текстов x 50 шагов. В режиме raw это ~5 мин (как у боевого
# guidance), с проекцией добавляется kNN и локальный PCA на каждом шаге --
# примерно втрое дороже, ~15 мин. Плюс сбор банка (~2 мин), метрики (~2 мин) и
# загрузка (~1 мин). Итого до ~25 мин; лимит 4 ч с запасом на PROBE_NUM_TEXTS=5000.
#SBATCH --time=04:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Probe: guidance с проекцией на касательное (PROBE_GRAD_MODE=${PROBE_GRAD_MODE:-tangent}, PROBE_SCALE_MODE=${PROBE_SCALE_MODE:-adaptive}, ratio=${PROBE_TARGET_RATIO:-0.1})"

torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 probe_projected_guidance.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Probe finished."
