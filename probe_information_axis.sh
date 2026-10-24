#!/bin/bash
#SBATCH --job-name=probe_infoaxis
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Оценка: 1000 текстов x 50 шагов -- около 2 мин счета (guidance здесь без
# классификатора, только арифметика с целью), плюс сбор банка и k-means до
# 3 мин у режимов quantize/seqquant/rank, плюс ~4 мин на метрики (они
# считаются ДВАЖДЫ: GEN vs TRG и сама цель vs TRG) и ~1 мин на загрузку.
# Итого ~12 мин; лимит 3 ч с запасом.
#SBATCH --time=03:00:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

export WANDB_MODE=offline

source run_flags.sh

echo "Probe: ось информативности (PROBE_DEGRADE=${PROBE_DEGRADE:-prefix}, PROBE_LEVEL=${PROBE_LEVEL:-8}, ratio=${PROBE_TARGET_RATIO:-0.1})"

torchrun --master_port=$((20000 + SLURM_JOB_ID % 10000)) --nproc_per_node=1 probe_information_axis.py \
    --scheduler sd \
    --encoder_name bert-base-cased \
    --swap_cfg_coef 0.0 \
    --project_name pgwtd \
    ${ARCH_FLAGS} ${DATA_FLAGS} \
    --seed 0

echo "Probe finished."
