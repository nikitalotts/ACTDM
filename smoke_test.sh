#!/bin/bash
# Короткий проверочный прогон пайплайна перед боевым обучением.
#
# Гоняет те же кодовые пути, что и полное обучение (данные -> обучение ->
# чекпоинт -> генерация -> метрики), но на 200 шагах и 200 генерируемых
# текстах. Все артефакты получают суффикс -smoke и НЕ пересекаются с боевыми:
# после проверки их можно просто удалить.
#
#   ./smoke_test.sh decoder    безусловный декодер (нужен diffuseq)   ~15 мин
#   ./smoke_test.sh diffuseq   диффузия diffuseq + генерация          ~30 мин
#   ./smoke_test.sh gpt        GPT2-бейзлайн + генерация              ~30 мин
#   ./smoke_test.sh eval       отдельный прогон eval по smoke-чекпоинтам
#
#   ./smoke_test.sh timing-diffuseq   замер скорости шага на 1 GPU с боевыми
#   ./smoke_test.sh timing-gpt        параметрами; падает по таймауту, artefacts
#                                     уходят в *-timing1gpu, боевые не трогает
#
# gpt ни от чего не зависит -- можно пускать сразу.
# diffuseq требует посчитанных статистик (./run_wikipedia.sh stats) и
# smoke-декодера, поэтому: decoder -> diffuseq.
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
        ARCH_TYPE=unconditional sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_decoder train_decoder.sh
        echo "==> ждем datasets/wikipedia/decoder-*-smoke.pth, затем: ./smoke_test.sh diffuseq"
        ;;
    diffuseq)
        ARCH_TYPE=diffuseq sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_diffuseq train_diffusion.sh
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
        #   gpt      -- микробатч 256 на 4 карты = 64 на карту (батч 512
        #               набирается накоплением, на память оно не влияет)
        # Число шагов тоже разное: у gpt tqdm считает МИКРОшаги.
        AT="${1#timing-}"
        if [ "${AT}" = "gpt" ]; then
            SCRIPT=train_gpt2.sh
            DEFAULT_BATCH=64
            TOTAL_STEPS="100 000 микрошагов"
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
        ARCH_TYPE=gpt sbatch --time=${SMOKE_TIME} \
            --job-name=smoke_eval_gpt eval_gpt2.sh
        ;;
    *)
        sed -n '2,20p' "$0"
        exit 1
        ;;
esac
