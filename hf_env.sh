# Единое место, где задаются пути к кэшам HuggingFace. Подключается всеми
# точками входа: run_flags.sh (все sbatch-задания) и run_wikipedia.sh
# (стадия data на login-ноде). Тот же дефолт продублирован в
# prefetch_offline.py -- прогрев и чтение обязаны смотреть в один каталог.
#
# Раскладка на Харизме (маркер кластера -- наличие /scratch/$USER):
#   HF_HOME -> home (~13 GB: модели, токенизаторы, скрипты метрик).
#       Scratch чистят ежемесячно, а без этих кэшей eval падает уже после
#       дней обучения -- держим там, где переживут чистку.
#   HF_DATASETS_CACHE -> scratch (~30 GB: сырой кэш датасетов).
#       Нужен только на стадии data и восстановим повторным скачиванием;
#       в home он не влезает.
# Вне кластера ничего не трогаем. Заданные снаружи значения не перекрываются.
if [ -d "/scratch/${USER:-}" ]; then
    [ -z "${HF_HOME:-}" ] && export HF_HOME="${HOME}/hf_cache"
    [ -z "${HF_DATASETS_CACHE:-}" ] && export HF_DATASETS_CACHE="/scratch/${USER}/hf_datasets"
fi
echo "HF_HOME=${HF_HOME:-<не задан: ~/.cache/huggingface>}  HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-<не задан: HF_HOME/datasets>}"
