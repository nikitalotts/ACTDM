import ml_collections
import os
import re
from transformers import PretrainedConfig, AutoConfig

from utils.schemes import (
    ARCHITECTURE_TYPES, SPLIT_SCHEMES, check_split_scheme, default_split_scheme,
)


# --- бюджеты обучения -----------------------------------------------------------
# Эффективный батч у ВСЕХ подходов одинаковый -- 512. У диффузий он такой же,
# как был в ВКР; у gpt поднят со 128 до 512, чтобы в таблице гиперпараметров
# статьи батч не отличался между подходами.
#
# Число шагов остается разным (150k у диффузий, 50k у gpt) и подобрано так, как
# было в ВКР: диффузии выходят на плато позже, gpt переставал улучшаться после
# ~10k шагов. Обе модели все равно останавливаются по сходимости, и берется
# лучший чекпоинт по tracked_metric.
#
# per_gpu -- сколько примеров карта тянет за микрошаг; подобрано замером на
# V100-32GB (find_max_batch.py, предел: gpt 82, diffuseq 426, genie 629,
# unconditional 960). Где 512 не влезает за один шаг, добирается накоплением.
WORLD_SIZE = 4
TRAINING_RECIPE = {
    "gpt": {
        # Эффективный батч поднят со 128 (как было в ВКР) до 512 -- такого же,
        # как у диффузий: формально одинаковый батч снимает вопрос о
        # сопоставимости. В память 512 за один шаг не влезает (предел 82 на
        # карту), поэтому 64 x 4 карты x 2 шага накопления.
        # 64 на карту -- это 81% памяти по замеру, плюс буферы DDP; если smoke
        # покажет OOM, безопасная замена -- per_gpu 32 с накоплением 4
        # (49% памяти, тот же эффективный батч, на 5% медленнее).
        "effective_batch": 512,
        "optimizer_steps": 50_000,
        "eval_every": 2_500,
        "per_gpu": 64,
    },
    "default": {                      # genie, diffuseq, guidance, unconditional
        "effective_batch": 512,
        "optimizer_steps": 150_000,
        "eval_every": 12_500,
        "per_gpu": 128,
    },
}


def training_budget(training, architecture_type):
    """Раскладывает рецепт на батч, накопление и число микрошагов."""
    r = TRAINING_RECIPE.get(architecture_type, TRAINING_RECIPE["default"])
    micro_batch = r["per_gpu"] * WORLD_SIZE
    accum = max(1, r["effective_batch"] // micro_batch)

    training.accum_batch_steps = accum
    training.batch_size = micro_batch
    training.training_iters = r["optimizer_steps"] * accum
    training.checkpoint_freq = r["eval_every"] * accum
    training.eval_freq = r["eval_every"] * accum
    return training


def create_config(args):
    """Собирает конфиг под architecture_type.

    gpt -- авторегрессионная модель, у нее нет диффузии, декодера и классификатора,
    поэтому конфиг собирается отдельной веткой. Общими остаются датасеты,
    длины последовательностей и метрики.
    """
    architecture_type = getattr(args, "architecture_type", ARCHITECTURE_TYPES[0])
    if architecture_type not in ARCHITECTURE_TYPES:
        raise Exception(
            f"Unknown architecture_type: {architecture_type}. "
            f"Expected one of {ARCHITECTURE_TYPES}"
        )
    if architecture_type == "gpt":
        return create_gpt_config(args)

    config = ml_collections.ConfigDict()

    config.work_dir = os.getcwd()
    
    training = config.training = ml_collections.ConfigDict()
    training_budget(training, architecture_type)
    training.ode_sampling = False
    training.checkpoints_folder = f"{config.work_dir}/checkpoints/"
    training.checkpoint_name = ""

    
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5000 * training.accum_batch_steps
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 1000
    validation.num_gen_texts = 5000
    validation.texts_path = f"{config.work_dir}/generated_texts"
    validation.cfg_coef = 0.
    validation.classifier_guidance_scale = args.classifier_guidance_scale

    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.solver = 'euler'
    dynamic.scheduler = args.scheduler
    dynamic.N = 50
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.ode_sampling = False
    dynamic.coef_d = args.coef_d

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.downstream_task = ""
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.encoder_name = args.encoder_name

    if "bert" in model.encoder_name.lower():
        model.encoder_link = 'google-bert/bert-base-cased'
    elif "roberta" in model.encoder_name.lower():
        model.encoder_link = 'FacebookAI/roberta-base'
    elif "t5" in model.encoder_name.lower():
        model.encoder_link = 'google-t5/t5-base'
    elif "bart" in model.encoder_name.lower():
        model.encoder_link = 'facebook/bart-base'

    model.conditional_encoder_name = model.encoder_name
    model.encoder_name_hash = model.encoder_name.replace("/", "-")
    model.conditional_encoder_name_hash = model.conditional_encoder_name.replace("/", "-")

    data = config.data = ml_collections.ConfigDict()
    data.datasets = create_datasets_config(args)
    data.base_path = f"{config.work_dir}/datasets"
    data.max_sequence_len = get_sequence_len(data.datasets.datasets_list[0])
    data.max_context_len = get_context_len(data.datasets.datasets_list[0])
    data.path = ""
    data.swap_cfg_coef = args.swap_cfg_coef
    data.enc_gen_mean = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-mean.pt"
    data.enc_gen_std = f"{data.base_path}/{data.datasets.datasets_list[0]}/statistics/encodings-{model.encoder_name_hash}-std.pt"

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.use_self_cond = True
    config.emb = args.emb
    config.mode = args.mode

    # --- режим работы -----------------------------------------------------------
    # Всё поведение выводится из architecture_type, отдельных флагов режима нет:
    #
    #   genie          -- условная диффузия, промпт через cross-attention
    #   diffuseq       -- условная диффузия, промпт через latent replacement
    #   guidance       -- безусловная диффузия + classifier guidance при генерации
    #   unconditional  -- безусловная диффузия, промпт не используется
    config.architecture_type = architecture_type

    # условна ли сама диффузия (видит ли denoising network промпт)
    config.is_conditional = config.architecture_type in ("genie", "diffuseq")
    # cross-attention нужен только genie
    config.use_cross_attention = config.architecture_type == "genie"
    # конкатенация промпта с зашумленным продолжением -- только diffuseq
    config.use_latent_replacement = config.architecture_type == "diffuseq"
    # градиент классификатора на генерации -- только guidance
    config.classifier_guidance = config.architecture_type == "guidance"
    # нужен ли промпт пайплайну: в guidance он нужен классификатору,
    # хотя в саму диффузию не подается. Это НЕ то же самое, что is_conditional.
    config.is_pipeline_conditional = config.architecture_type != "unconditional"

    config.guidance_scale = args.classifier_guidance_scale
    if config.classifier_guidance and config.guidance_scale <= 0:
        raise Exception(
            "architecture_type=guidance требует --classifier_guidance_scale > 0"
        )
    if not config.classifier_guidance and config.guidance_scale != 0:
        print(
            "[CONFIG] WARNING: --classifier_guidance_scale задан при "
            f"architecture_type={config.architecture_type}, guidance не будет "
            "применяться, scale сброшен в 0"
        )
        config.guidance_scale = 0.
        validation.classifier_guidance_scale = 0.

    # Нормализация энкодингов статистиками датасета (EncNormalizer).
    # По умолчанию включена. При --emb нормализация идет по статистикам словаря
    # внутри Encoder и этим флагом не управляется.
    config.normalize_encodings = not args.no_normalize_encodings and not config.emb
    dataset_name = data.datasets.datasets_list[0]
    data.split_scheme = args.split_scheme or default_split_scheme(dataset_name)
    check_split_scheme(dataset_name, data.split_scheme)

    decoder = config.decoder = create_decoder_config()
    decoder.dataset = data.datasets.datasets_list[0]
    decoder.name = f"decoder-{model.encoder_name_hash}-128-transformer"
    decoder.name += decoder.suffix
    decoder.is_conditional = config.use_cross_attention
    if decoder.is_conditional:
        decoder.name += "-conditional"
    if config.emb:
        decoder.name += "-emb"
    # декодер обязан жить в том же пространстве латентов и на той же нарезке данных
    decoder.name += artifact_suffix(config)
    decoder.decoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{decoder.name}.pth"
    if decoder.max_sequence_len < data.max_sequence_len:
        raise Exception("Decoder max_sequence_len is less than required")

    cond_encoder = config.cond_encoder = create_cond_encoder_config()
    cond_encoder.dataset = data.datasets.datasets_list[0]
    # Классификатор токенизирует src/trg длинами данных (max_context_len /
    # max_sequence_len) -- ровно та геометрия, что на guidance-инференсе.
    # Длины входят в имя: классификатор, обученный старым кодом с фиксированной
    # шириной 80, не должен молча переиспользоваться
    cond_encoder.name = f"conditional-encoder-{model.encoder_name_hash}-{data.max_context_len}x{data.max_sequence_len}-transformer"
    cond_encoder.name += cond_encoder.suffix
    cond_encoder.mode = config.mode
    if cond_encoder.empty_trg_prob > 0:
        cond_encoder.name += f'-empty_trg_prob={cond_encoder.empty_trg_prob}'
    cond_encoder.name += f'-epochs-{cond_encoder.epochs}'
    # схема негативов входит в имя: иначе три схемы обучения писали бы
    # классификатор в один и тот же файл и затирали друг друга
    cond_encoder.augmentation_scheme = args.augmentation_scheme
    cond_encoder.name += f'-{cond_encoder.augmentation_scheme}'
    # масштаб времени не меняет форму весов, но меняет смысл обученной модели,
    # поэтому варианты должны лежать в разных файлах
    cond_encoder.time_scale = float(getattr(args, 'time_scale', 1.0))
    if cond_encoder.time_scale != 1.0:
        cond_encoder.name += f'-ts{cond_encoder.time_scale:g}'
    cond_encoder.name += artifact_suffix(config)
    cond_encoder.cond_encoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{cond_encoder.name}.pth"
    cond_encoder.use_conditional_encoder = config.classifier_guidance

    config.se_config = create_se_config()
    config.se_config.is_conditional = config.is_conditional
    config.se_config.is_decoder = config.use_cross_attention
    config.se_config.vocab_size = AutoConfig.from_pretrained(model.encoder_link).vocab_size
    config.se_config.use_self_cond = config.use_self_cond

    # Метрики: в безусловном режиме оценивается качество текста как такового,
    # в условных и в guidance -- соответствие промпту.
    data.datasets.metrics[dataset_name] = metrics_for_mode(config.is_pipeline_conditional)

    print(f"[CONFIG] architecture_type={config.architecture_type}")
    print(f"[CONFIG] is_conditional={config.is_conditional} "
          f"(cross-attention: {'ON' if config.use_cross_attention else 'OFF'}, "
          f"latent replacement: {'ON' if config.use_latent_replacement else 'OFF'})")
    print(f"[CONFIG] classifier_guidance={config.classifier_guidance}, scale={config.guidance_scale}")
    print(f"[CONFIG] normalize_encodings={config.normalize_encodings}, emb={config.emb}")
    print(f"[CONFIG] split_scheme={data.split_scheme}, augmentation_scheme={cond_encoder.augmentation_scheme}")
    if config.classifier_guidance:
        print(f"[CONFIG] cond_encoder.time_scale={cond_encoder.time_scale}")

    config.project_name = args.project_name
    config.timesteps = "linear"
    pref = "emb" if config.emb else "tencdm"
    training.checkpoints_prefix = f"{pref}-{model.encoder_name_hash}-{training.batch_size}-{optim.lr}-{data.datasets.datasets_list[0]}-cfg={data.swap_cfg_coef}"
    training.checkpoints_prefix += checkpoints_prefix_suffix(config)
    config.eval = args.eval or False
    
    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = True
    config.save_top_k = 5
    return apply_smoke_overrides(config)


def create_gpt_config(args):
    """Конфиг авторегрессионного GPT-2, обучаемого с нуля.

    Датасеты, длины и метрики берутся из тех же общих функций, что и у диффузии,
    чтобы обе ветки работали на одних и тех же данных.
    """
    config = ml_collections.ConfigDict()

    config.work_dir = os.getcwd()

    training = config.training = ml_collections.ConfigDict()
    training_budget(training, "gpt")
    training.ode_sampling = False
    training.checkpoints_folder = f"{config.work_dir}/checkpoints/"
    training.checkpoint_name = ""

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    # scheduler.step_update получает номер ОПТИМИЗАТОРНОГО шага (см. gpt2_holder),
    # поэтому прогрев задается в них же. 2000 -- как на rocstories в дипломе
    optim.linear_warmup = 2000
    # В ВКР было 1e-4 при эффективном батче 128. Батч поднят до 512 (чтобы
    # совпадал с диффузионным), и lr масштабирован по правилу корня:
    # 1e-4 * sqrt(512/128) = 2e-4.
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 32
    validation.num_gen_texts = 5000
    validation.texts_path = f"{config.work_dir}/generated_texts"
    validation.cfg_coef = 0.

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.encoder_name = "gpt2-medium"
    # Этим токенизатором dataset_wiki режет текст на промпт/продолжение.
    # Берется тот же BERT, что у диффузии: граница разбиения обязана совпадать
    # во всех подходах, иначе gpt сравнивается на других парах
    model.encoder_link = 'google-bert/bert-base-cased'

    data = config.data = ml_collections.ConfigDict()
    data.datasets = create_datasets_config(args)
    data.base_path = f"{config.work_dir}/datasets"
    data.max_sequence_len = get_sequence_len(data.datasets.datasets_list[0])
    data.max_context_len = get_context_len(data.datasets.datasets_list[0])
    data.path = ""
    data.swap_cfg_coef = 0.0
    dataset_name = data.datasets.datasets_list[0]
    data.split_scheme = args.split_scheme or default_split_scheme(dataset_name)
    check_split_scheme(dataset_name, data.split_scheme)

    config.architecture_type = "gpt"
    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.emb = False
    config.use_self_cond = False

    # GPT авторегрессионный: он всегда обусловлен промптом, но у него нет
    # ни cross-attention, ни latent replacement, ни classifier guidance
    config.is_conditional = True
    config.use_cross_attention = False
    config.use_latent_replacement = False
    config.classifier_guidance = False
    config.guidance_scale = 0.
    config.is_pipeline_conditional = True
    config.normalize_encodings = False

    data.datasets.metrics[dataset_name] = metrics_for_mode(True)

    config.project_name = args.project_name
    training.checkpoints_prefix = f"gpt2-medium-512-{optim.lr}-{data.datasets.datasets_list[0]}"
    training.checkpoints_prefix += artifact_suffix(config)
    config.eval = args.eval or False

    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = True
    config.save_top_k = 2

    print(f"[CONFIG] architecture_type=gpt (GPT2-medium from scratch)")
    print(f"[CONFIG] dataset={data.datasets.datasets_list[0]}, split_scheme={data.split_scheme}")
    print(f"[CONFIG] max_context_len={data.max_context_len}, max_sequence_len={data.max_sequence_len}")
    print(f"[CONFIG] training_iters={training.training_iters}, batch_size={training.batch_size}, "
          f"accum_batch_steps={training.accum_batch_steps}")

    return apply_smoke_overrides(config)


def data_budget(config):
    """Сколько данных модель увидит за прогон.

    Для честного сравнения подходов в статье это ключевая величина: модели
    должны видеть данные одинаковое число раз. Считается одинаково для
    диффузии и gpt, у которых разный accum_batch_steps:

        training_iters -- число МИКРОшагов (у gpt их в accum раз больше),
        batch_size     -- примеров на микрошаг суммарно по всем GPU.
    """
    micro_steps = config.training.training_iters
    accum = config.training.accum_batch_steps
    return {
        "examples_seen": config.training.batch_size * micro_steps,
        "effective_batch": config.training.batch_size * accum,
        "optimizer_steps": micro_steps // accum,
    }


def print_data_budget(config):
    b = data_budget(config)
    print(f"[CONFIG] бюджет обучения: {b['examples_seen'] / 1e6:.1f} млн примеров "
          f"(эффективный батч {b['effective_batch']}, "
          f"{b['optimizer_steps']} оптимизаторных шагов)")
    return config


def apply_env_overrides(config):
    """Пометка прогона (RUN_TAG) и подмена батча (BATCH_SIZE) из окружения.

    Нужны для замерочных запусков: прогнать боевые параметры на другом числе
    GPU, посмотреть реальную скорость шага и упасть по таймауту, ничего при
    этом не записав в боевые каталоги. RUN_TAG уходит в checkpoints_prefix,
    поэтому чекпоинты и сгенерированные тексты такого прогона лежат отдельно,
    а декодер и статистики переиспользуются боевые (они только читаются).
    """
    tag = os.environ.get("RUN_TAG", "").strip()
    batch = os.environ.get("BATCH_SIZE", "").strip()

    # Прогон с другим батчем -- это другой прогон, его чекпоинты не должны
    # лежать под боевым именем. Метку не требуем: подставляем сами.
    if batch and not tag:
        tag = f"bs{batch}"

    if tag:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "-", tag)
        config.training.checkpoints_prefix += f"-{safe}"
        print(f"[CONFIG] RUN_TAG={safe}: checkpoints_prefix={config.training.checkpoints_prefix}")

    if batch:
        config.training.batch_size = int(batch)
        print(f"[CONFIG] BATCH_SIZE={batch} (глобальный батч подменен)")

    return config


def apply_smoke_overrides(config):
    """Короткий проверочный прогон всего пайплайна: SMOKE=1 в окружении.

    Задача -- за десятки минут прогнать те же кодовые пути, что и боевое
    обучение (данные -> обучение -> чекпоинт -> генерация -> метрики), но на
    сотнях шагов вместо сотен тысяч.

    Все артефакты уводятся в имена с суффиксом -smoke: иначе тестовый прогон
    перезапишет боевые чекпоинты, а load_checkpoint боевого запуска подхватит
    недоученные веса из smoke-прогона.
    """
    config = apply_env_overrides(config)

    if os.environ.get("SMOKE", "0") != "1":
        return print_data_budget(config)

    accum = config.training.accum_batch_steps
    config.training.training_iters = 200 * accum
    config.training.eval_freq = 100 * accum
    config.training.checkpoint_freq = 100 * accum
    # прогрев на 5000 шагов при 200 шагах обучения оставил бы lr около нуля.
    # Задается в оптимизаторных шагах, поэтому на accum не умножается
    config.optim.linear_warmup = 20

    # генерация и метрики -- самая долгая часть eval, для проверки хватает
    # пары сотен текстов
    config.validation.num_gen_texts = 200
    config.validation.batch_size = min(config.validation.batch_size, 50)

    config.training.checkpoints_prefix += "-smoke"

    artifacts_dir = f"{config.data.base_path}/{config.data.datasets.datasets_list[0]}"

    if "decoder" in config:
        config.decoder.max_train_steps = 200
        config.decoder.name += "-smoke"
        config.decoder.decoder_path = f"{artifacts_dir}/{config.decoder.name}.pth"

    # классификатор guidance переименовываем тоже: иначе короткий прогон
    # затер бы боевой чекпоинт классификатора недоученными весами
    if "cond_encoder" in config:
        config.cond_encoder.epochs = 1
        config.cond_encoder.name += "-smoke"
        config.cond_encoder.cond_encoder_path = f"{artifacts_dir}/{config.cond_encoder.name}.pth"

    print(f"[CONFIG] SMOKE=1: training_iters={config.training.training_iters}, "
          f"eval_freq={config.training.eval_freq}, "
          f"num_gen_texts={config.validation.num_gen_texts}, "
          f"checkpoints_prefix={config.training.checkpoints_prefix}")
    return print_data_budget(config)


def metrics_for_mode(is_pipeline_conditional):
    """В условных режимах меряем соответствие промпту, в безусловном -- качество текста."""
    if is_pipeline_conditional:
        return {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        }
    return {
        "metrics": ["mauve", "div", "ppl"],
        "tracked_metric": "mauve",
    }


def artifact_suffix(config):
    """Суффикс, общий для декодера, классификатора и чекпоинтов диффузии.

    Все они обязаны жить в одном пространстве латентов и на одной нарезке данных,
    поэтому несовпадающие варианты не должны попадать в один файл. Значения по
    умолчанию дают пустой суффикс -- старые имена остаются валидными.
    """
    suffix = ""
    # у gpt нет латентного пространства, нормализация к нему неприменима
    if config.architecture_type != "gpt" and not config.normalize_encodings and not config.emb:
        suffix += "-unnorm"
    dataset_name = config.data.datasets.datasets_list[0]
    if config.data.split_scheme != default_split_scheme(dataset_name):
        suffix += f"-{config.data.split_scheme}"
    return suffix


def checkpoints_prefix_suffix(config):
    """Суффикс имени чекпоинта диффузии.

    guidance и unconditional используют ОДНУ И ТУ ЖЕ безусловную диффузию
    (guidance -- это та же модель плюс классификатор на генерации), поэтому
    у них общий суффикс и чекпоинт переиспользуется. У genie суффикс пустой --
    имена остаются обратно совместимыми.
    """
    by_arch = {
        "genie": "",
        "diffuseq": "-diffuseq",
        "guidance": "-uncond",
        "unconditional": "-uncond",
    }
    return by_arch[config.architecture_type] + artifact_suffix(config)


def create_se_config():
    se_config = AutoConfig.from_pretrained("bert-base-cased")
    se_config.attention_head_size = se_config.hidden_size / se_config.num_attention_heads
    return se_config


def create_datasets_config(args):
    config = ml_collections.ConfigDict()
    config.downstream_tasks = ["qqp", "xsum", "paradetox", "wiki_auto", "rocstories"]
    if args.dataset_name is None:
        config.datasets_list = ["rocstories"]
    else:
        config.datasets_list = [args.dataset_name]
    config.metrics = {
        "rocstories": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "wikipedia": {"metrics": ["mauve", "div", "ppl"],
                       "tracked_metric": "mauve"},
        "qqp": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "xsum": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
        "wiki_auto": {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        },
    }
    return config


def create_decoder_config():
    config = ml_collections.ConfigDict()

    config.max_sequence_len = 80
    config.noise_sigma = 0.2
    config.lr = 1e-4
    config.betas = (0.9, 0.98)
    config.weight_decay = 0.001
    config.batch_size = 64
    config.epochs = 1
    # ограничение числа шагов обучения; None -- полная эпоха (см. SMOKE=1)
    config.max_train_steps = None
    config.max_norm = 1.0
    config.is_conditional = False
    config.dataset = ""
    config.T = 0.15
    config.eps = 0.001
    config.diffusion_forward = True
    config.suffix = ""
    config.num_hidden_layers = 3
    
    return config


def create_cond_encoder_config():
    config = ml_collections.ConfigDict()
    config.dataset = ""
    config.batch_size = 32
    config.max_norm = 5.0
    config.epochs = 13
    config.lr = 1e-4
    config.weight_decay = 0.01
    config.betas = (0.9, 0.98)
    config.max_sequence_len = 80
    config.suffix = ""
    config.T = 1.0
    config.eps = 0.001
    config.empty_trg_prob = 0.0
    # масштаб непрерывного t перед синусоидальным эмбеддингом;
    # фактическое значение приходит из --time_scale, см. create_config
    config.time_scale = 1.0

    return config


def get_sequence_len(dataset_name):
    # для wikipedia схема prefix LM: 128 токенов суммарно, поровну на промпт
    # и продолжение, то есть 64 + 64
    data = {
        "wikipedia": 64,
        "rocstories": 35, 
        "qqp": 50,
        "xsum": 64,
        "wiki_auto": 100,
    }
    return data[dataset_name]


def get_context_len(dataset_name):
    data = {
        "wikipedia": 64,
        "rocstories": 45, 
        "qqp": 50,
        "xsum": 512,
        "wiki_auto": 100,
    }
    return data[dataset_name]

    
