"""Названия схем, общие для конфига, парсера аргументов и data/load.py.

Модуль намеренно не тянет torch: его импортирует data/load.py, которому
для скачивания датасета нужны только имена схем.
"""

# --- как условие связано с диффузией -------------------------------------------
# Первая в списке -- значение по умолчанию.
ARCHITECTURE_TYPE_HELP = {
    "genie": "условная диффузия, промпт через cross-attention в denoising network",
    "diffuseq": "условная диффузия, промпт через latent replacement "
                "(латенты промпта фиксируются на каждом шаге)",
    "guidance": "безусловная диффузия + classifier guidance при генерации",
    "unconditional": "безусловная диффузия, промпт не используется вообще",
    "gpt": "авторегрессионный GPT-2 вместо диффузии, обучается с нуля на парах промпт/продолжение",
}
ARCHITECTURE_TYPES = list(ARCHITECTURE_TYPE_HELP)

# --- схемы разбиения текста на промпт/продолжение --------------------------------
# Первая в списке -- значение по умолчанию (историческое поведение).
# Схемы rocstories режут историю по предложениям, схемы wikipedia -- по токенам.
SPLIT_SCHEME_HELP = {
    "half": "rocstories: предложения 1,2,3 -> 4,5, одна пара на историю",
    "last_sentence": "rocstories: предложения 1-4 -> предложение 5, одна пара на историю",
    "sliding": "rocstories: 1->2,3 | 1,2->3,4 | 1,2,3->4,5, три пары на историю",
    "prefix_lm": "wikipedia: первые 50%% токенов -- промпт, остальные -- продолжение "
                 "(схема prefix LM из TESS-2, граница фиксирована)",
    "random_prefix": "wikipedia: промпт -- случайная доля токенов до 50%% "
                     "(прежнее поведение, граница плавает)",
}
SPLIT_SCHEMES = list(SPLIT_SCHEME_HELP)

# Схема задает и то, как данные нарезаны на диске, и то, как их читает загрузчик,
# поэтому несовместимую с датасетом пару надо отсекать в конфиге, а не ловить
# потом по странным метрикам.
SPLIT_SCHEMES_BY_DATASET = {
    "rocstories": ["half", "last_sentence", "sliding"],
    "wikipedia": ["prefix_lm", "random_prefix"],
}
DEFAULT_SPLIT_SCHEME = {
    "rocstories": "half",
    "wikipedia": "prefix_lm",
}


def default_split_scheme(dataset_name):
    """Схема по умолчанию для датасета.

    Для downstream-задач (qqp, xsum, wiki_auto) разбиение задано самим датасетом,
    поэтому там схема не используется.
    """
    for key, scheme in DEFAULT_SPLIT_SCHEME.items():
        if key in dataset_name:
            return scheme
    return SPLIT_SCHEMES[0]


def check_split_scheme(dataset_name, split_scheme):
    """Бросает исключение, если схема не применима к датасету."""
    for key, allowed in SPLIT_SCHEMES_BY_DATASET.items():
        if key in dataset_name and split_scheme not in allowed:
            raise Exception(
                f"split_scheme={split_scheme} не применим к датасету {dataset_name}. "
                f"Допустимые схемы: {allowed}"
            )

# --- схемы генерации негативов для классификатора (classifier guidance) ---------
AUGMENTATION_SCHEME_HELP = {
    "shuffled": "негативы -- перемешанные пары промпт/продолжение (ВКР, схема 1)",
    "augmented": "негативы те же перемешанные, но все продолжения (и позитивы, "
                 "и негативы) проходят диффузионную реконструкцию x_0 (ВКР, схема 2)",
    "combined": "shuffled + второй тип негативов: диффузионная реконструкция "
                "настоящего продолжения (ВКР, схема 3)",
}
AUGMENTATION_SCHEMES = list(AUGMENTATION_SCHEME_HELP)
