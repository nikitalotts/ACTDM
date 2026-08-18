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
        # отдельный каталог, чтобы боевой прогон их не подхватил; BATCH_SIZE=128
        # держит на единственной GPU ту же нагрузку, что приходится на одну GPU
        # в четырехкарточном прогоне (512/4). Поэтому измеренное s/it -- это
        # сразу время шага боевого прогона, делить на 4 НЕ надо.
        AT="${1#timing-}"
        [ "${AT}" = "gpt" ] && SCRIPT=train_gpt2.sh || SCRIPT=train_diffusion.sh
        SMOKE=0 RUN_TAG=timing1gpu NPROC=1 BATCH_SIZE="${BATCH_SIZE:-128}" \
            ARCH_TYPE="${AT}" sbatch \
            --gpus-per-task=1 --time="${TIMING_TIME:-1:00:00}" \
            --job-name=timing-${AT} "${SCRIPT}"
        echo "==> смотрите темп в slurm_logs/<jobid>-timing-${AT}.log (строка вида '1.59s/it')"
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
