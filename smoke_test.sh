#!/bin/bash
# Короткий проверочный прогон пайплайна перед боевым обучением.
#
# Гоняет те же кодовые пути, что и полное обучение (данные -> обучение ->
# чекпоинт -> генерация -> метрики), но на 200 шагах и 200 генерируемых
# текстах. Все артефакты получают суффикс -smoke и НЕ пересекаются с боевыми:
# после проверки их можно просто удалить.
#
#   ./smoke_test.sh decoder    smoke-декодеры ~15 мин; НУЖЕН, только если
#                              боевые decoder-*.pth еще не обучены -- иначе
#                              diffuseq/genie сами возьмут боевой декодер
#   ./smoke_test.sh diffuseq   диффузия diffuseq + генерация          ~30 мин
#   ./smoke_test.sh genie      диффузия genie + генерация             ~30 мин
#   ./smoke_test.sh gpt        GPT2-бейзлайн + генерация              ~30 мин
#   ./smoke_test.sh eval       отдельный прогон eval по smoke-чекпоинтам
#
#   ./smoke_test.sh timing-diffuseq   замер скорости шага на 1 GPU с боевыми
#   ./smoke_test.sh timing-gpt        параметрами; падает по таймауту, artefacts
#                                     уходят в *-timing1gpu, боевые не трогает
#
# gpt ни от чего не зависит -- можно пускать сразу.
# diffuseq и genie требуют посчитанных статистик (./run_wikipedia.sh stats) и
# декодера. Декодер берется так: если есть smoke-декодер -- он, иначе боевой
# (диффузия его только читает, а проверяться на боевом даже честнее -- это тот
# самый файл, который возьмет полный прогон). Стадия decoder нужна лишь тогда,
# когда боевых декодеров еще нет.
#
# Удалить следы проверки:
#   rm -rf checkpoints/*-smoke datasets/wikipedia/*-smoke.pth generated_texts/*-smoke

set -e
export DATASET=wikipedia
export SMOKE=1

mkdir -p "$(dirname "$0")/slurm_logs"
source "$(dirname "$0")/hf_env.sh"

# smoke-заданиям не нужны боевые лимиты времени
SMOKE_TIME="${SMOKE_TIME:-2:00:00}"

case "$1" in
    decoder)
        # genie декодирует с cross-attention на промпт -- ему нужен свой
        # декодер; diffuseq и unconditional делят безусловный.
        #
        # Стадия нужна, ТОЛЬКО пока боевых декодеров нет: когда они обучены,
        # diffuseq/genie подставят их сами. Обучение декодера при этом всегда
        # пишет в -smoke файл (флаг TRAINING_DECODER в model/train_decoder.py),
        # так что боевой декодер этой стадией не затрется.
        ARCH_TYPE=unconditional sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_decoder-unconditional train_decoder.sh
        ARCH_TYPE=genie sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_decoder-genie train_decoder.sh
        echo "==> ждем оба datasets/wikipedia/decoder-*-smoke.pth, затем: ./smoke_test.sh diffuseq | genie"
        ;;
    diffuseq)
        ARCH_TYPE=diffuseq sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_diffuseq train_diffusion.sh
        ;;
    genie)
        # у genie свой код: условие идет через cross-attention, а декодер
        # условный. С diffuseq этот путь не пересекается, и первый раз он
        # отрабатывает только на eval -- то есть через часы боевого прогона.
        ARCH_TYPE=genie sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_genie train_diffusion.sh
        ;;
    gpt)
        ARCH_TYPE=gpt sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_gpt train_gpt2.sh
        ;;
    timing-diffuseq|timing-gpt)
        # Замер скорости шага на 1 GPU с БОЕВЫМИ параметрами (150k/200k шагов,
        # тот же батч на GPU, что в реальном прогоне). Задание намеренно падает
        # по --time: нужен только темп из tqdm, а не результат.
        #
        # SMOKE=0 -- параметры настоящие; RUN_TAG уводит чекпоинты и тексты в
        # отдельный каталог, чтобы боевой прогон их не подхватил.
        #
        # BATCH_SIZE держит на единственной карте РОВНО ту нагрузку, что
        # приходится на одну карту в четырехкарточном прогоне, поэтому
        # измеренное s/it -- сразу время шага боевого прогона, делить на 4
        # не надо. Значения разные, потому что разный микробатч:
        #   диффузия -- батч 512 на 4 карты = 128 на карту
        #   gpt      -- микробатч 128 на 4 карты = 32 на карту (батч 512
        #               набирается накоплением, на память оно не влияет)
        # Число шагов тоже разное: у gpt tqdm считает МИКРОшаги.
        AT="${1#timing-}"
        if [ "${AT}" = "gpt" ]; then
            SCRIPT=train_gpt2.sh
            DEFAULT_BATCH=32
            TOTAL_STEPS="200 000 микрошагов"
        else
            SCRIPT=train_diffusion.sh
            DEFAULT_BATCH=128
            TOTAL_STEPS="150 000 шагов"
        fi
        SMOKE=0 RUN_TAG=timing1gpu NPROC=1 BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH}}" \
            ARCH_TYPE="${AT}" sbatch \
            --gpus-per-task=1 --time="${TIMING_TIME:-1:00:00}" \
            --job-name=timing-${AT} "${SCRIPT}"
        echo "==> темп смотрите в slurm_logs/<jobid>-timing-${AT}.log (строка вида '1.59s/it')"
        echo "==> оценка боевого прогона: s/it x ${TOTAL_STEPS}"
        echo "==> задание упадет по таймауту -- это ожидаемо"
        ;;
    eval)
        ARCH_TYPE=diffuseq sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_eval_diffuseq eval_diffusion.sh
        ARCH_TYPE=genie sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_eval_genie eval_diffusion.sh
        ARCH_TYPE=gpt sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_eval_gpt eval_gpt2.sh
        ;;
    *)
        sed -n '2,29p' "$0"
        exit 1
        ;;
esac
