# Единое место, где задается путь к кэшу HuggingFace (модели, токенизаторы,
# скрипты метрик). Подключается всеми точками входа: run_flags.sh (все
# sbatch-задания) и run_wikipedia.sh (стадия data на login-ноде).
# Тот же дефолт продублирован в prefetch_offline.py -- прогрев и чтение
# обязаны смотреть в один и тот же каталог.
#
# Логика: если HF_HOME не задан снаружи и на машине есть scratch (Харизма) --
# кэш живет на scratch (NVMe, восстановим повторным прогревом). Иначе ничего
# не трогаем: остается стандартный ~/.cache/huggingface или то, что задал
# пользователь.
if [ -z "${HF_HOME:-}" ] && [ -d "/scratch/${USER}" ]; then
    export HF_HOME="/scratch/${USER}/hf_cache"
fi
echo "HF_HOME=${HF_HOME:-<не задан: ~/.cache/huggingface>}"
