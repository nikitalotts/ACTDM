#!/bin/bash
#SBATCH --job-name=train_decoder
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
# Часа не хватает: перед обучением задание токенизирует весь train-сплит
# (1.5 млн абзацев), и только потом идет эпоха примерно на 23к шагов
#SBATCH --time=8:00:00

source ~/.bashrc
eval "$(conda shell.bash hook)"

module purge
module load Python

conda deactivate
conda activate pgwtd

# на compute-нодах нет интернета: без offline wandb.init виснет на ретраях
export WANDB_MODE=offline

source run_flags.sh

echo "Starting script..."

python -m model.train_decoder --encoder_name='bert-base-cased' --project_name='pgwtd' ${ARCH_FLAGS} ${DATA_FLAGS}
 
echo "Script finished."