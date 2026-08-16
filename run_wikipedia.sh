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
# Боевой прогон никогда не должен оказаться коротким проверочным: sbatch
# наследует окружение целиком, и SMOKE=1, случайно оставшийся в шелле
# (например, после `source smoke_test.sh`), молча урезал бы обучение до
# 200 шагов. Гасим явно -- smoke запускается только через smoke_test.sh.
export SMOKE=0

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
        ARCH_TYPE=genie         sbatch --job-name=train_diffusion-genie         train_diffusion.sh
        ARCH_TYPE=diffuseq      sbatch --job-name=train_diffusion-diffuseq      train_diffusion.sh
        ARCH_TYPE=unconditional sbatch --job-name=train_diffusion-unconditional train_diffusion.sh
        echo "==> когда обучится unconditional: ./run_wikipedia.sh classifiers"
        ;;
    gpt)
        ARCH_TYPE=gpt sbatch train_gpt2.sh
        ;;
    classifiers)
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
        sed -n '2,24p' "$0"
        exit 1
        ;;
esac
