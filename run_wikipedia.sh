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
#   ./run_wikipedia.sh gpt          GPT2-medium-бейзлайн, 355M     (после data)
#   ./run_wikipedia.sh gpt-small    GPT2-small, 124M: бейзлайн, сопоставимый
#                                   с диффузиями по числу параметров (102M у
#                                   diffuseq, 130M у genie). fp16 и без
#                                   пересчёта активаций -- ему хватает памяти
#   ./run_wikipedia.sh classifiers  3 классификатора для guidance  (после
#                                   unconditional: augmented и combined
#                                   реконструируют x_0 ее чекпоинтом)
#   ./run_wikipedia.sh classifier-shuffled  только схема shuffled -- она строит
#                                   негативы перестановкой пар и диффузию не
#                                   трогает, поэтому идет сразу после data,
#                                   параллельно чему угодно
#   ./run_wikipedia.sh guidance     оценка ТОЛЬКО guidance, три схемы
#                                   (после classifiers). Отдельная стадия нужна,
#                                   потому что eval переоценивает и уже
#                                   посчитанные genie/diffuseq/uncond/gpt
#   ./run_wikipedia.sh eval         финальная оценка всех подходов (после всего)
#   ./run_wikipedia.sh stat         статистическая оценка: NUM_SEEDS прогонов
#                                   на подход, mean +- std и 95% CI. Шесть
#                                   заданий -- genie, diffuseq, guidance x3, gpt.
#                                   NUM_SEEDS=20 по умолчанию,
#                                   CHECKPOINT_STEP=87500 -- шаг сравнения.
#   ./run_wikipedia.sh sweep-gpt    свип декодинга авторегрессионного
#                                   бейзлайна: greedy плюс сэмплинг с разными
#                                   температурой/top_p/top_k. Нужен, чтобы в
#                                   статье сравнивать с ЛУЧШИМ gpt, а не со
#                                   случайно выбранным декодингом
#   ./run_wikipedia.sh stat-uncond  то же для unconditional, отдельно: у него
#                                   метрики mauve/div/ppl, он не попадает в одну
#                                   таблицу с условными, и считается дольше всех
#
# Порядок целиком: data -> stats -> decoders -> diffusion -> classifiers -> eval,
# gpt можно пускать сразу после data параллельно всему остальному.
#
# guidance отдельно не обучается: он использует чекпоинт безусловной диффузии
# (общий checkpoints_prefix) плюс классификатор на генерации.

set -e
export DATASET=wikipedia
export PATH="$HOME/anaconda3/condabin:$PATH"
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

# CHECKPOINT_STEP пинит файл, который грузит restore_parameters. Для eval это
# то, что нужно, а вот обучение он заставил бы ДОЗАПУСКАТЬСЯ с указанного шага.
# Поэтому запоминаем и гасим: стадии stat выставят его себе сами.
_CKPT_STEP="${CHECKPOINT_STEP:-}"
unset CHECKPOINT_STEP

# сюда пишут stdout+stderr все sbatch-задания: slurm_logs/<jobid>-<имя>.log;
# slurm не создает каталог сам, без него задание умирает молча
mkdir -p "$(dirname "$0")/slurm_logs"

# кэш HuggingFace: стадия data качает датасет напрямую отсюда (не через
# sbatch), поэтому настройка кэша нужна и здесь, а не только в run_flags.sh
source "$(dirname "$0")/hf_env.sh"

# Проект Slurm: на Харизме sbatch требует -A <проект> и дефолта не имеет, а
# стадии зовут sbatch сами, изнутри. Значение по умолчанию -- тот же проект,
# по которому считает projstat.sh. Переопределяется извне:
#   SBATCH_ACCOUNT=proj_1795 ./run_wikipedia.sh gpt
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-proj_1891}"


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
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate /home/amshabalin/anaconda3/envs/pgwtd
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
    gpt-small)
        # Зачем второй бейзлайн: у gpt2-medium 355M против 102M у diffuseq и
        # 130M у genie, и при равном числе увиденных примеров он тратит в
        # 3.5-5 раз больше FLOPs. Первое, что скажет рецензент, если medium
        # выигрывает. gpt2-small -- 124M, ровно между диффузиями.
        #
        # fp16 -- то же, чем обучен medium (на V100 он втрое быстрее bf16).
        # Пересчёт активаций выключен: он нужен был, чтобы на medium влез батч
        # 128 на карту, у small памяти с запасом, а стоит он около четверти
        # скорости. Перед запуском проверить пробником, что батч действительно
        # влезает:
        #   GPT_MODEL=gpt2 GRAD_CKPT=0 AMP_DTYPE=fp16 ARCHS=gpt MAX_BATCH=128 \
        #       sbatch find_max_batch.sh
        #
        # Лимит времени -- не формальность: кластер проверяет квоту прямо на
        # sbatch и отказывает, если (карты x время) больше остатка проекта.
        # 75 ч из train_gpt2.sh это 300 GPU-часов, и при остатке меньше этого
        # задание не отправится вообще.
        #
        # СКОЛЬКО НА САМОМ ДЕЛЕ НУЖНО. Время прогона это НЕ шаг x число шагов:
        # поверх лежат примерно 8 часов накладных -- препроцессинг шарда при
        # каждой перезагрузке (около 27 раз за прогон, по 12-15 минут) плюс
        # eval-ы. Величина почти не зависит от размера модели: у medium те же
        # 8 ч при 36.9 ч счёта, у small -- при 12.2 ч. Считать надо
        # СЛОЖЕНИЕМ, а не коэффициентом:
        #     small:  87500 x 0.50 c = 12.2 ч + 8 ч = ~20 ч
        #     medium: 87500 x 1.52 c = 36.9 ч + 8 ч = ~45 ч
        # Первый прогон small с лимитом 20 ч был снят на шаге 79865 из 87500
        # именно поэтому.
        #
        # Дефолт оставлен 20 ч намеренно: свежему прогону нужно ~26, но это
        # 104 GPU-часа, а под конец проекта столько проверка квоты не
        # пропустит. Значит один дозапуск штатен -- он стоит ~45 мин на
        # повторный препроцессинг и продолжается с last.pth (каждые 2500
        # шагов). Дозапуск: TIME_LIMIT=6:00:00 ./run_wikipedia.sh gpt-small
        # Остаток квоты: bash projstat.sh proj_1891
        GPT_MODEL=gpt2 \
        AMP_DTYPE="${AMP_DTYPE:-fp16}" \
        GRAD_CKPT="${GRAD_CKPT:-0}" \
        ARCH_TYPE=gpt sbatch --time="${TIME_LIMIT:-20:00:00}" \
            --job-name=train_gpt2-small train_gpt2.sh
        echo "==> каталог чекпоинтов: checkpoints/gpt2-512-0.0002-wikipedia"
        echo "==> medium не затрагивается: у размеров разные префиксы"
        ;;
    classifier-shuffled)
        # Единственная схема, не зависящая от диффузии: негативы получаются
        # перестановкой пар промпт/продолжение внутри батча, чекпоинт
        # безусловной модели ей не нужен (в отличие от augmented и combined,
        # которые реконструируют x_0). Значит ее можно обучать параллельно
        # с самой диффузией, не дожидаясь ее конца.
        sbatch train_conditional_encoder_shuffled.sh
        echo "==> augmented и combined -- отдельно, после unconditional:        ./run_wikipedia.sh classifiers"
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
    guidance)
        # guidance = чекпоинт безусловной диффузии + градиент классификатора на
        # генерации, обучать нечего. Три схемы отличаются только тем, каким
        # классификатором ведут, и пишут разные json -- имена берут
        # augmentation_scheme, поэтому друг друга не затирают.
        #
        # Стадия отдельная от eval: там переоцениваются и genie/diffuseq/uncond,
        # у которых результаты уже посчитаны, и запускается gpt, который может
        # быть еще не обучен.
        for AUG in shuffled augmented combined; do
            CLS=$(ls datasets/wikipedia/conditional-encoder-*-"${AUG}".pth 2>/dev/null | head -1)
            if [ -z "${CLS}" ]; then
                echo "Нет классификатора схемы ${AUG}: datasets/wikipedia/conditional-encoder-*-${AUG}.pth" >&2
                echo "Классификаторы обучаются раньше: ./run_wikipedia.sh classifiers" >&2
                exit 1
            fi
        done
        for AUG in shuffled augmented combined; do
            ARCH_TYPE=guidance AUG_SCHEME=${AUG} sbatch \
                --job-name=eval_diffusion-guidance-${AUG} eval_diffusion.sh
        done
        echo "==> сила guidance по умолчанию CG_SCALE=10.0, поменять: CG_SCALE=5.0 ./run_wikipedia.sh guidance"
        echo "==> метрики в логах slurm, тексты в generated_texts/<checkpoints_prefix>/"
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
        echo "==> для std и 95% CI по сидам: ./run_wikipedia.sh stat"
        ;;
    sweep-gpt)
        # У gpt есть температура и top_p, у диффузий таких ручек нет. Если взять
        # один наугад выбранный декодинг, рецензент справедливо скажет, что
        # бейзлайн ослаблен. Свип дешёвый -- одна карта, минуты на конфигурацию --
        # и закрывает вопрос: в статью идёт ЛУЧШИЙ gpt.
        export CHECKPOINT_STEP="${_CKPT_STEP:-87500}"
        echo "==> CHECKPOINT_STEP=${CHECKPOINT_STEP}"
        ARCH_TYPE=gpt DECODING=greedy \
            sbatch --job-name=sweep-gpt-greedy eval_gpt2.sh
        for T in 0.7 0.8 0.9 1.0; do
            ARCH_TYPE=gpt DECODING=sampling TEMPERATURE=${T} TOP_P=0.95 TOP_K=0 \
                sbatch --job-name=sweep-gpt-t${T} eval_gpt2.sh
        done
        ARCH_TYPE=gpt DECODING=sampling TEMPERATURE=1.0 TOP_P=0.9 TOP_K=0 \
            sbatch --job-name=sweep-gpt-p0.9 eval_gpt2.sh
        ARCH_TYPE=gpt DECODING=sampling TEMPERATURE=1.0 TOP_P=1.0 TOP_K=50 \
            sbatch --job-name=sweep-gpt-k50 eval_gpt2.sh
        echo "==> семь конфигураций, каждая пишет свой json (режим входит в имя)"
        echo "==> собрать метрики:"
        echo "==>   for f in slurm_logs/*sweep-gpt*.log; do echo \"== \$f\"; grep -aE '^(bleu|bert-score|rouge1|rouge2|rougeL):' \"\$f\" | tail -5; done"
        ;;
    stat|stat-uncond)
        # Статистическая оценка для статьи: NUM_SEEDS независимых прогонов
        # генерации, из них mean +- std и 95% CI.
        #
        # Шаг сравнения задается ЯВНО. Без этого restore_parameters берет
        # max(нумерованных), а он разный: genie и unconditional доехали до
        # 100000, diffuseq и gpt остановлены на 87500 -- подходы сравнивались бы
        # на разном бюджете, и в логе это выглядело бы нормально.
        export CHECKPOINT_STEP="${_CKPT_STEP:-87500}"
        export NUM_SEEDS="${NUM_SEEDS:-20}"
        echo "==> CHECKPOINT_STEP=${CHECKPOINT_STEP}  NUM_SEEDS=${NUM_SEEDS}"
        echo "==> пилот перед боевым прогоном: NUM_SEEDS=3 ./run_wikipedia.sh $1"

        if [ "$1" = "stat-uncond" ]; then
            # Отдельной стадией: у unconditional метрики mauve/div/ppl, а не
            # bleu/rouge, то есть в одну таблицу с условными подходами он не
            # ставится. И считается он дольше всех -- mauve и ppl примерно вдвое
            # дороже bleu/rouge. Пусть его стоимость будет видна отдельно.
            ARCH_TYPE=unconditional sbatch --time="${TIME_LIMIT:-16:00:00}" --job-name=stat-unconditional eval_diffusion_stat.sh
            echo "==> unconditional: mauve/div/ppl -- это АБСОЛЮТНЫЕ числа,"
            echo "==> референс по ним не считается (см. заметку в RUNBOOK)"
        else
            ARCH_TYPE=genie    sbatch --time="${TIME_LIMIT:-12:00:00}" --job-name=stat-genie    eval_diffusion_stat.sh
            ARCH_TYPE=diffuseq sbatch --time="${TIME_LIMIT:-12:00:00}" --job-name=stat-diffuseq eval_diffusion_stat.sh
            for AUG in shuffled augmented combined; do
                ARCH_TYPE=guidance AUG_SCHEME=${AUG} \
                    sbatch --time="${TIME_LIMIT:-12:00:00}" --job-name=stat-guidance-${AUG} eval_diffusion_stat.sh
            done
            ARCH_TYPE=gpt sbatch --time="${TIME_LIMIT:-12:00:00}" --job-name=stat-gpt eval_gpt2_stat.sh
            echo "==> шесть заданий: genie, diffuseq, guidance x3, gpt"
            echo "==> unconditional отдельно: ./run_wikipedia.sh stat-uncond"
        fi
        echo "==> сверить, что все подходы считаны на одном шаге:"
        echo "==>   grep -h -A4 'STATISTICAL EVAL SUMMARY' slurm_logs/*stat*.log | grep -E 'checkpoint|step'"
        ;;
    *)
        # показать шапку с описанием стадий
        sed -n '2,52p' "$0"
        exit 1
        ;;
esac
