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
