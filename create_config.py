import ml_collections
import os
from transformers import PretrainedConfig, AutoConfig

from utils.schemes import ARCHITECTURE_TYPES, SPLIT_SCHEMES


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
    training.accum_batch_steps = 1
    training.training_iters = 150_000 * training.accum_batch_steps
    training.checkpoint_freq = 12_500 * training.accum_batch_steps 
    training.eval_freq = 12_500 
    training.batch_size = 512 // training.accum_batch_steps
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

    if 'wikipedia' in data.datasets.datasets_list and config.is_pipeline_conditional:
        raise Exception(
            "wikipedia -- безусловный датасет, для него доступен только "
            "architecture_type=unconditional"
        )

    # Нормализация энкодингов статистиками датасета (EncNormalizer).
    # По умолчанию включена. При --emb нормализация идет по статистикам словаря
    # внутри Encoder и этим флагом не управляется.
    config.normalize_encodings = not args.no_normalize_encodings and not config.emb
    data.split_scheme = args.split_scheme

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
    cond_encoder.name = f"conditional-encoder-{model.encoder_name_hash}-{cond_encoder.max_sequence_len}-transformer"
    cond_encoder.name += cond_encoder.suffix
    if cond_encoder.max_sequence_len < data.max_sequence_len:
        raise Exception("Conditional Encoder max_sequence_len is less than required")
    cond_encoder.mode = config.mode
    if cond_encoder.empty_trg_prob > 0:
        cond_encoder.name += f'-empty_trg_prob={cond_encoder.empty_trg_prob}'
    cond_encoder.name += f'-epochs-{cond_encoder.epochs}'
    # схема негативов входит в имя: иначе три схемы обучения писали бы
    # классификатор в один и тот же файл и затирали друг друга
    cond_encoder.augmentation_scheme = args.augmentation_scheme
    cond_encoder.name += f'-{cond_encoder.augmentation_scheme}'
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
    if config.is_pipeline_conditional:
        data.datasets.metrics["rocstories"] = {
            "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
            "tracked_metric": "bert-score",
        }
    else:
        data.datasets.metrics["rocstories"] = {
            "metrics": ["mauve", "div", "ppl"],
            "tracked_metric": "mauve",
        }

    print(f"[CONFIG] architecture_type={config.architecture_type}")
    print(f"[CONFIG] is_conditional={config.is_conditional} "
          f"(cross-attention: {'ON' if config.use_cross_attention else 'OFF'}, "
          f"latent replacement: {'ON' if config.use_latent_replacement else 'OFF'})")
    print(f"[CONFIG] classifier_guidance={config.classifier_guidance}, scale={config.guidance_scale}")
    print(f"[CONFIG] normalize_encodings={config.normalize_encodings}, emb={config.emb}")
    print(f"[CONFIG] split_scheme={data.split_scheme}, augmentation_scheme={cond_encoder.augmentation_scheme}")

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
    return config


def create_gpt_config(args):
    """Конфиг авторегрессионного GPT-2, обучаемого с нуля.

    Датасеты, длины и метрики берутся из тех же общих функций, что и у диффузии,
    чтобы обе ветки работали на одних и тех же данных.
    """
    config = ml_collections.ConfigDict()

    config.work_dir = os.getcwd()

    training = config.training = ml_collections.ConfigDict()
    training.accum_batch_steps = 4
    training.training_iters = 50_000 * training.accum_batch_steps
    training.checkpoint_freq = 2_500 * training.accum_batch_steps
    training.eval_freq = 2_500 * training.accum_batch_steps
    training.batch_size = 128 // training.accum_batch_steps
    training.ode_sampling = False
    training.checkpoints_folder = f"{config.work_dir}/checkpoints/"
    training.checkpoint_name = ""

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 500 * training.accum_batch_steps
    optim.lr = 1e-4
    optim.min_lr = 1e-4
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

    data = config.data = ml_collections.ConfigDict()
    data.datasets = create_datasets_config(args)
    data.base_path = f"{config.work_dir}/datasets"
    data.max_sequence_len = get_sequence_len(data.datasets.datasets_list[0])
    data.max_context_len = get_context_len(data.datasets.datasets_list[0])
    data.path = ""
    data.swap_cfg_coef = 0.0
    data.split_scheme = args.split_scheme

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

    if 'wikipedia' in data.datasets.datasets_list:
        raise Exception(
            "wikipedia -- безусловный датасет, architecture_type=gpt требует пары "
            "промпт/продолжение"
        )

    data.datasets.metrics["rocstories"] = {
        "metrics": ["bleu", "bert-score", "rouge1", "rouge2", "rougeL"],
        "tracked_metric": "bert-score",
    }

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

    return config


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
    if config.data.split_scheme != SPLIT_SCHEMES[0]:
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

    return config


def get_sequence_len(dataset_name):
    data = {
        "wikipedia": 128,
        "rocstories": 35, 
        "qqp": 50,
        "xsum": 64,
        "wiki_auto": 100,
    }
    return data[dataset_name]


def get_context_len(dataset_name):
    data = {
        "wikipedia": 128,
        "rocstories": 45, 
        "qqp": 50,
        "xsum": 512,
        "wiki_auto": 100,
    }
    return data[dataset_name]

    
