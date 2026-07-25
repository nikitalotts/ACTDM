# Общие флаги запуска для всех скриптов подпроекта.
#
# Использование:  source mode_flags.sh    затем  ... ${MODE_FLAGS} ${DATA_FLAGS}
#
# --- режим работы (MODE_FLAGS) ---
# MODE=unconditional        -- безусловная диффузия
# MODE=conditional          -- условная диффузия (промпт через cross-attention)
# MODE=classifier_guidance  -- безусловная диффузия + classifier guidance на генерации
# Для classifier_guidance силу задает CG_SCALE.
#
# --- данные и пространство латентов (DATA_FLAGS) ---
# SPLIT_SCHEME=last_sentence|sliding|half  -- нарезка истории rocstories на промпт/продолжение.
#                                             Должна совпадать с той, с которой скачан датасет.
# NORMALIZE=1|0                            -- нормализовать ли энкодинги статистиками датасета.
#
# DATA_FLAGS обязаны совпадать у декодера, диффузии и классификатора: они задают
# и нарезку данных, и пространство латентов. Имена чекпоинтов это учитывают.

MODE="${MODE:-unconditional}"
CG_SCALE="${CG_SCALE:-10.0}"
SPLIT_SCHEME="${SPLIT_SCHEME:-last_sentence}"
NORMALIZE="${NORMALIZE:-1}"

case "${MODE}" in
    unconditional)
        MODE_FLAGS=""
        ;;
    conditional)
        MODE_FLAGS="--is_conditional"
        ;;
    classifier_guidance)
        MODE_FLAGS="--classifier_guidance --classifier_guidance_scale=${CG_SCALE}"
        ;;
    *)
        echo "Unknown MODE='${MODE}'. Expected: unconditional | conditional | classifier_guidance" >&2
        exit 1
        ;;
esac

case "${SPLIT_SCHEME}" in
    last_sentence|sliding|half)
        DATA_FLAGS="--split_scheme ${SPLIT_SCHEME}"
        ;;
    *)
        echo "Unknown SPLIT_SCHEME='${SPLIT_SCHEME}'. Expected: last_sentence | sliding | half" >&2
        exit 1
        ;;
esac

case "${NORMALIZE}" in
    1) ;;
    0) DATA_FLAGS="${DATA_FLAGS} --no_normalize_encodings" ;;
    *)
        echo "Unknown NORMALIZE='${NORMALIZE}'. Expected: 1 | 0" >&2
        exit 1
        ;;
esac

echo "MODE=${MODE}  MODE_FLAGS='${MODE_FLAGS}'"
echo "SPLIT_SCHEME=${SPLIT_SCHEME}  NORMALIZE=${NORMALIZE}  DATA_FLAGS='${DATA_FLAGS}'"
