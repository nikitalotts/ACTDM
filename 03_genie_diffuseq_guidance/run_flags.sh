# Общие флаги запуска для всех скриптов подпроекта.
#
# Использование:  source run_flags.sh    затем  ... ${ARCH_FLAGS} ${DATA_FLAGS}
#
# --- архитектура (ARCH_FLAGS) ---
# ARCH_TYPE=genie          -- условная диффузия, промпт через cross-attention
# ARCH_TYPE=diffuseq       -- условная диффузия, промпт через latent replacement
# ARCH_TYPE=guidance       -- безусловная диффузия + classifier guidance на генерации
# ARCH_TYPE=unconditional  -- безусловная диффузия, промпт не используется
# Для guidance силу задает CG_SCALE.
#
# --- данные и пространство латентов (DATA_FLAGS) ---
# SPLIT_SCHEME=half|last_sentence|sliding  -- нарезка истории rocstories на промпт/продолжение.
#                                             Должна совпадать с той, с которой скачан датасет.
# NORMALIZE=1|0                            -- нормализовать ли энкодинги статистиками датасета.
#
# DATA_FLAGS обязаны совпадать у декодера, диффузии и классификатора: они задают
# и нарезку данных, и пространство латентов. Имена артефактов это учитывают.

ARCH_TYPE="${ARCH_TYPE:-genie}"
CG_SCALE="${CG_SCALE:-10.0}"
SPLIT_SCHEME="${SPLIT_SCHEME:-half}"
NORMALIZE="${NORMALIZE:-1}"

case "${ARCH_TYPE}" in
    genie|diffuseq|unconditional)
        ARCH_FLAGS="--architecture_type ${ARCH_TYPE}"
        ;;
    guidance)
        ARCH_FLAGS="--architecture_type guidance --classifier_guidance_scale=${CG_SCALE}"
        ;;
    *)
        echo "Unknown ARCH_TYPE='${ARCH_TYPE}'. Expected: genie | diffuseq | guidance | unconditional" >&2
        exit 1
        ;;
esac

case "${SPLIT_SCHEME}" in
    half|last_sentence|sliding)
        DATA_FLAGS="--split_scheme ${SPLIT_SCHEME}"
        ;;
    *)
        echo "Unknown SPLIT_SCHEME='${SPLIT_SCHEME}'. Expected: half | last_sentence | sliding" >&2
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

echo "ARCH_TYPE=${ARCH_TYPE}  ARCH_FLAGS='${ARCH_FLAGS}'"
echo "SPLIT_SCHEME=${SPLIT_SCHEME}  NORMALIZE=${NORMALIZE}  DATA_FLAGS='${DATA_FLAGS}'"
