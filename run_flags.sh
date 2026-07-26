# Общие флаги запуска для всех скриптов подпроекта.
#
# Использование:  source run_flags.sh    затем  ... ${ARCH_FLAGS} ${DATA_FLAGS}
#
# --- архитектура (ARCH_FLAGS) ---
# ARCH_TYPE=genie          -- условная диффузия, промпт через cross-attention
# ARCH_TYPE=diffuseq       -- условная диффузия, промпт через latent replacement
# ARCH_TYPE=guidance       -- безусловная диффузия + classifier guidance на генерации
# ARCH_TYPE=unconditional  -- безусловная диффузия, промпт не используется
# ARCH_TYPE=gpt            -- авторегрессионный GPT-2 вместо диффузии
# Для guidance силу задает CG_SCALE.
#
# --- данные и пространство латентов (DATA_FLAGS) ---
# DATASET=rocstories|wikipedia   -- на чем учимся.
# SPLIT_SCHEME=<схема>           -- нарезка текста на промпт/продолжение.
#     rocstories: half (по умолчанию) | last_sentence | sliding -- по предложениям;
#     wikipedia:  prefix_lm (по умолчанию) | random_prefix      -- по токенам.
#     Если не задана, берется схема по умолчанию для датасета.
# NORMALIZE=1|0                  -- нормализовать ли энкодинги статистиками датасета.
#
# DATA_FLAGS обязаны совпадать у декодера, диффузии и классификатора: они задают
# и нарезку данных, и пространство латентов. Имена артефактов это учитывают.

ARCH_TYPE="${ARCH_TYPE:-genie}"
CG_SCALE="${CG_SCALE:-10.0}"
DATASET="${DATASET:-rocstories}"
SPLIT_SCHEME="${SPLIT_SCHEME:-}"
NORMALIZE="${NORMALIZE:-1}"

case "${ARCH_TYPE}" in
    genie|diffuseq|unconditional|gpt)
        ARCH_FLAGS="--architecture_type ${ARCH_TYPE}"
        ;;
    guidance)
        ARCH_FLAGS="--architecture_type guidance --classifier_guidance_scale=${CG_SCALE}"
        ;;
    *)
        echo "Unknown ARCH_TYPE='${ARCH_TYPE}'. Expected: genie | diffuseq | guidance | unconditional | gpt" >&2
        exit 1
        ;;
esac

case "${DATASET}" in
    rocstories|wikipedia)
        DATA_FLAGS="--dataset_name ${DATASET}"
        ;;
    *)
        echo "Unknown DATASET='${DATASET}'. Expected: rocstories | wikipedia" >&2
        exit 1
        ;;
esac

# схему передаем только если она задана явно -- иначе конфиг возьмет
# значение по умолчанию для выбранного датасета
case "${SPLIT_SCHEME}" in
    "")
        ;;
    half|last_sentence|sliding|prefix_lm|random_prefix)
        DATA_FLAGS="${DATA_FLAGS} --split_scheme ${SPLIT_SCHEME}"
        ;;
    *)
        echo "Unknown SPLIT_SCHEME='${SPLIT_SCHEME}'. Expected: half | last_sentence |" \
             "sliding | prefix_lm | random_prefix" >&2
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
echo "DATASET=${DATASET}  SPLIT_SCHEME=${SPLIT_SCHEME:-<по умолчанию>}  NORMALIZE=${NORMALIZE}  DATA_FLAGS='${DATA_FLAGS}'"
