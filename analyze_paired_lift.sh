#!/bin/bash
#SBATCH --job-name=paired_lift
#SBATCH --output=slurm_logs/%j-%x.log
#SBATCH --error=slurm_logs/%j-%x.log
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --nodes=1
# GPU НЕ ЗАПРАШИВАЕТСЯ: счет идет по уже записанным текстам, это униграммный F1
# на 1000 примеров -- секунды чистого питона. Если очередь откажется принимать
# задание без карты, пускать через LIFT_GPU=1 (см. run_wikipedia.sh probes-lift):
# аргумент командной строки sbatch перебивает директивы файла.
#SBATCH --time=00:10:00

source ~/.bashrc
module purge
module load Python
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/amshabalin/anaconda3/envs/pgwtd

# Шаблон задает, какие прогоны сравниваются. По умолчанию -- свип по alpha
# стадии probes-snap на том же датасете и той же силе шага: смешивать прогоны
# с разным ratio нельзя, парное сравнение требует одних и тех же примеров.
LIFT_GLOB="${LIFT_GLOB:-probe_results/projected_guidance-${DATASET:-wikipedia}-*snap-pos-a*-r${PROBE_TARGET_RATIO:-0.005}-*.json}"

shopt -s nullglob
FILES=( ${LIFT_GLOB} )
if [ ${#FILES[@]} -eq 0 ]; then
    echo "НЕ НАЙДЕНО ни одного json по шаблону: ${LIFT_GLOB}" >&2
    echo "Что лежит в probe_results:" >&2
    ls -1 probe_results/ 2>/dev/null | head -40 >&2
    exit 1
fi

echo "Парная статистика по ${#FILES[@]} прогонам (шаблон ${LIFT_GLOB}):"
for f in "${FILES[@]}"; do echo "  ${f}"; done
echo ""

OUT="probe_results/paired_lift-${DATASET:-wikipedia}.txt"
set -o pipefail
python analyze_paired_lift.py "${FILES[@]}" | tee "${OUT}"
echo ""
echo "Та же таблица лежит в ${OUT}"
echo "Paired lift finished."
