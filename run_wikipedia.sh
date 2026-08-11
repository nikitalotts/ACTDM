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
        python -m data.load --dataset_name wikipedia \
            ${NUM_TEXTS:+--num_texts ${NUM_TEXTS}}
        echo "==> обучение читает только datasets/wikipedia; сырой кэш можно удалить:"
        echo "==>   rm -rf ~/.cache/huggingface/datasets/wikimedia___wikipedia*"
        echo "==> дальше: ./run_wikipedia.sh stats   (и параллельно gpt)"
        ;;
    stats)
        sbatch make_statistics.sh
        echo "==> когда появятся datasets/wikipedia/statistics/*.pt: ./run_wikipedia.sh decoders"
        ;;
    decoders)
        # genie декодирует с cross-attention на промпт -- ему нужен свой декодер;
        # diffuseq, guidance и unconditional делят безусловный
        ARCH_TYPE=genie         sbatch train_decoder.sh
        ARCH_TYPE=unconditional sbatch train_decoder.sh
        echo "==> когда появятся оба datasets/wikipedia/decoder-*.pth: ./run_wikipedia.sh diffusion"
        ;;
    diffusion)
        ARCH_TYPE=genie         sbatch train_diffusion.sh
        ARCH_TYPE=diffuseq      sbatch train_diffusion.sh
        ARCH_TYPE=unconditional sbatch train_diffusion.sh
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
        ARCH_TYPE=genie         sbatch eval_diffusion.sh
        ARCH_TYPE=diffuseq      sbatch eval_diffusion.sh
        ARCH_TYPE=unconditional sbatch eval_diffusion.sh
        for AUG in shuffled augmented combined; do
            ARCH_TYPE=guidance AUG_SCHEME=${AUG} sbatch eval_diffusion.sh
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
