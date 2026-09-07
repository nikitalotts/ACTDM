#!/bin/bash
# Прогон всех экспериментов статьи на wikipedia.
#
# Стадии зависят друг от друга через артефакты на диске (датасет, статистики,
# декодеры, чекпоинты диффузии), поэтому каждая запускается руками ПОСЛЕ того,
# как отработала предыдущая. Внутри одной стадии задания независимы и уходят
# в slurm параллельно.
#
#   ./run_wikipedia.sh data         скачать и нарезать датасет (нужен интернет;
#                                   для пробного прогона: NUM_TEXTS=200000)
#   ./run_wikipedia.sh stats        статистики энкодера            (после data)
#   ./run_wikipedia.sh decoders     условный + безусловный декодер (после stats)
#   ./run_wikipedia.sh diffusion    диффузии genie/diffuseq/uncond (после decoders)
#   ./run_wikipedia.sh diffuseq     то же, но по одной модели за раз:
#   ./run_wikipedia.sh genie        genie | diffuseq | unconditional (=uncond)
#   ./run_wikipedia.sh unconditional
#   ./run_wikipedia.sh gpt          GPT2-бейзлайн                  (после data)
#   ./run_wikipedia.sh classifiers  3 классификатора для guidance  (после diffusion:
#                                   augmented и combined реконструируют x_0
#                                   чекпоинтом безусловной диффузии)
#   ./run_wikipedia.sh eval         финальная оценка всех подходов (после всего)
#
# Порядок целиком: data -> stats -> decoders -> diffusion -> classifiers -> eval,
# gpt можно пускать сразу после data параллельно всему остальному.
#
# guidance отдельно не обучается: он использует чекпоинт безусловной диффузии
# (общий checkpoints_prefix) плюс классификатор на генерации.

set -e
export DATASET=wikipedia
# Боевой прогон обязан быть боевым. sbatch наследует окружение целиком, поэтому
# любая переменная проверочных/замерочных режимов, случайно оставшаяся в шелле
# (например, после `source smoke_test.sh`), молча испортила бы обучение:
#   SMOKE=1      -- урезало бы обучение до 200 шагов
#   RUN_TAG=...  -- чекпоинты ушли бы в чужой каталог
#   BATCH_SIZE=. -- обучение шло бы с другим батчем
#   NPROC=1      -- четырехкарточное задание считало бы на одной GPU
# Гасим все явно: эти режимы запускаются только через smoke_test.sh.
export SMOKE=0
unset RUN_TAG BATCH_SIZE NPROC

# сюда пишут stdout+stderr все sbatch-задания: slurm_logs/<jobid>-<имя>.log;
# slurm не создает каталог сам, без него задание умирает молча
mkdir -p "$(dirname "$0")/slurm_logs"

# кэш HuggingFace: стадия data качает датасет напрямую отсюда (не через
# sbatch), поэтому настройка кэша нужна и здесь, а не только в run_flags.sh
source "$(dirname "$0")/hf_env.sh"

# сила guidance и масштаб времени классификатора на финальной оценке
CG_SCALE="${CG_SCALE:-10.0}"
TIME_SCALE="${TIME_SCALE:-1.0}"
export CG_SCALE TIME_SCALE

case "$1" in
    data)
        # нарезка на абзацы >=128 слов и train/valid/test 3000/7000;
        # разбиение на промпт/продолжение делается на лету при обучении.
        # Запускать на ноде с интернетом; перед первым запуском прогреть кэши
        # моделей: python prefetch_offline.py
        # Через sbatch эту стадию пускать нельзя: на compute-нодах нет интернета.
        # Активируем окружение сами -- иначе возьмется системный python2.7
        eval "$(conda shell.bash hook)" 2>/dev/null || true
        conda activate pgwtd
        python -m data.load --dataset_name wikipedia \
            ${NUM_TEXTS:+--num_texts ${NUM_TEXTS}}
        echo "==> обучение читает только datasets/wikipedia; сырой кэш можно удалить:"
        echo "==>   rm -rf ${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}/wikimedia___wikipedia*"
        echo "==> дальше: ./run_wikipedia.sh stats   (и параллельно gpt)"
        ;;
    stats)
        sbatch make_statistics.sh
        echo "==> когда появятся datasets/wikipedia/statistics/*.pt: ./run_wikipedia.sh decoders"
        ;;
    decoders)
        # genie декодирует с cross-attention на промпт -- ему нужен свой декодер;
        # diffuseq, guidance и unconditional делят безусловный
        # --job-name попадает в имя лога (%x в --output)
        ARCH_TYPE=genie         sbatch --job-name=train_decoder-genie         train_decoder.sh
        ARCH_TYPE=unconditional sbatch --job-name=train_decoder-unconditional train_decoder.sh
        echo "==> когда появятся оба datasets/wikipedia/decoder-*.pth: ./run_wikipedia.sh diffusion"
        ;;
    diffusion)
        # все три сразу; для запуска по одной есть отдельные стадии ниже
        for AT in genie diffuseq unconditional; do
            ARCH_TYPE=${AT} sbatch --job-name=train_diffusion-${AT} train_diffusion.sh
        done
        echo "==> когда обучится unconditional: ./run_wikipedia.sh classifiers"
        ;;
    genie|diffuseq|uncond|unconditional)
        # одна диффузия отдельным заданием -- когда нужно занять карты не всеми
        # тремя сразу или перезапустить только одну после обрыва.
        # uncond -- сокращение для unconditional, как и в smoke_test.sh:
        # два скрипта должны понимать одни и те же имена стадий
        AT="$1"
        [ "${AT}" = uncond ] && AT=unconditional
        ARCH_TYPE="${AT}" sbatch --job-name=train_diffusion-"${AT}" train_diffusion.sh
        ;;
    gpt)
        ARCH_TYPE=gpt sbatch train_gpt2.sh
        ;;
    classifiers)
        # Схемы augmented и combined реконструируют x_0 чекпоинтом безусловной
        # диффузии. Без него все три задания отстоят очередь, поднимут модели и
        # только тогда упадут по FileNotFoundError -- проверяем сразу здесь.
        if ! ls checkpoints/*-uncond/[0-9]*.pth >/dev/null 2>&1; then
            echo "Нет чекпоинта безусловной диффузии: checkpoints/*-uncond/<шаг>.pth" >&2
            echo "Классификаторы обучаются ПОСЛЕ нее: ./run_wikipedia.sh unconditional" >&2
            exit 1
        fi
        # каждая схема пишет свой файл conditional-encoder-*-64x64-*-<схема>*.pth
        sbatch train_conditional_encoder_shuffled.sh
        sbatch train_conditional_encoder_augmented.sh
        sbatch train_conditional_encoder_combined.sh
        echo "==> когда обучатся: ./run_wikipedia.sh eval"
        ;;
    eval)
        ARCH_TYPE=genie         sbatch --job-name=eval_diffusion-genie         eval_diffusion.sh
        ARCH_TYPE=diffuseq      sbatch --job-name=eval_diffusion-diffuseq      eval_diffusion.sh
        ARCH_TYPE=unconditional sbatch --job-name=eval_diffusion-unconditional eval_diffusion.sh
        for AUG in shuffled augmented combined; do
            ARCH_TYPE=guidance AUG_SCHEME=${AUG} sbatch --job-name=eval_diffusion-guidance-${AUG} eval_diffusion.sh
        done
        ARCH_TYPE=gpt sbatch eval_gpt2.sh
        echo "==> метрики в логах slurm, тексты в generated_texts/<checkpoints_prefix>/"
        echo "==> для std и 95% CI по сидам: те же ARCH_TYPE c eval_diffusion_stat.sh / eval_gpt2_stat.sh"
        ;;
    *)
        # показать шапку с описанием стадий
        sed -n '2,27p' "$0"
        exit 1
        ;;
esac
