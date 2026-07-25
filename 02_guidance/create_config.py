import ml_collections
import os
from transformers import PretrainedConfig, AutoConfig

from utils.schemes import SPLIT_SCHEMES


def create_config(args):
    config = ml_collections.ConfigDict()

    config.work_dir = os.getcwd()

    training = config.training = ml_collections.ConfigDict()
    training.accum_batch_steps = 1
    training.training_iters = 150_000 * training.accum_batch_steps
    training.training_iters = training.training_iters
    training.checkpoint_freq = 10_000 * training.accum_batch_steps
    training.eval_freq = 10_000 * training.accum_batch_steps
    training.batch_size = 128 // training.accum_batch_steps
    training.ode_sampling = False
    training.checkpoints_folder = f"{config.work_dir}/checkpoints/"
    training.checkpoint_name = ""

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5 * training.accum_batch_steps
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
    validation.batch_size = 256
    validation.num_gen_texts = args.num_diffusion_steps or 5000
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

    print('datasets', data.datasets.datasets_list)

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.use_self_cond = True
    # --- режим работы -----------------------------------------------------------
    # is_conditional        -- условная ли сама диффузия (cross-attention в denoising network)
    # classifier_guidance   -- применяется ли classifier guidance при генерации
    #
    # Ровно три допустимых сочетания:
    #   (False, False) -- безусловная диффузия
    #   (True,  False) -- условная диффузия
    #   (False, True)  -- безусловная диффузия + classifier guidance при генерации
    config.is_conditional = args.is_conditional or (
        False if 'rocstories' in data.datasets.datasets_list or 'wikipedia' in data.datasets.datasets_list else True)
    config.classifier_guidance = args.classifier_guidance
    config.guidance_scale = args.classifier_guidance_scale

    if config.is_conditional and config.classifier_guidance:
        raise Exception(
            "--is_conditional и --classifier_guidance взаимоисключающие: "
            "classifier guidance применяется поверх безусловной диффузии"
        )
    if config.classifier_guidance and config.guidance_scale <= 0:
        raise Exception(
            "--classifier_guidance требует --classifier_guidance_scale > 0"
        )
    if not config.classifier_guidance and config.guidance_scale != 0:
        print(
            "[CONFIG] WARNING: --classifier_guidance_scale задан без --classifier_guidance, "
            "guidance не будет применяться, scale сброшен в 0"
        )
        config.guidance_scale = 0.
        validation.classifier_guidance_scale = 0.

    if config.classifier_guidance:
        config.generation_mode = "classifier_guidance"
    elif config.is_conditional:
        config.generation_mode = "conditional"
    else:
        config.generation_mode = "unconditional"

    # Нужен ли промпт в пайплайне: в режимах 2 и 3 -- да, в режиме 1 -- нет.
    # Это НЕ то же самое, что is_conditional: в режиме 3 промпт нужен классификатору,
    # но в саму диффузию он не подается.
    config.is_pipeline_conditional = config.is_conditional or config.classifier_guidance

    config.emb = args.emb
    config.mode = args.mode

    # Нормализация энкодингов статистиками датасета (EncNormalizer).
    # По умолчанию включена -- как в исходном подпроекте.
    # При --emb нормализация идет по статистикам словаря внутри Encoder и этим флагом
    # не управляется, поэтому здесь она осмысленна только для режима без --emb.
    config.normalize_encodings = not args.no_normalize_encodings and not config.emb
    data.split_scheme = args.split_scheme

    decoder = config.decoder = create_decoder_config()
    decoder.dataset = data.datasets.datasets_list[0]
    decoder.name = f"decoder-{model.encoder_name_hash}-{config.decoder.max_sequence_len}-transformer"
    decoder.name += decoder.suffix
    decoder.is_conditional = False
    if decoder.is_conditional:
        decoder.name += "-conditional"
    if config.emb:
        decoder.name += "-emb"
    # Декодер обязан жить в том же пространстве, что и диффузия, поэтому
    # ненормализованный вариант хранится отдельным файлом
    decoder.name += artifact_suffix(config)
    decoder.decoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{decoder.name}.pth"
    if decoder.max_sequence_len < data.max_sequence_len:
        raise Exception("Decoder max_sequence_len is less than required")
    decoder.mode = config.mode

    cond_encoder = config.cond_encoder = create_cond_encoder_config()
    cond_encoder.dataset = data.datasets.datasets_list[0]
    cond_encoder.name = f"conditional-encoder-{model.encoder_name_hash}-{config.cond_encoder.max_sequence_len}-transformer"
    cond_encoder.name += cond_encoder.suffix
    if cond_encoder.max_sequence_len < data.max_sequence_len:
        raise Exception("Conditional Encoder max_sequence_len is less than required")
    cond_encoder.mode = config.mode
    if cond_encoder.empty_trg_prob > 0:
        cond_encoder.name += f'-empty_trg_prob={cond_encoder.empty_trg_prob}'
    cond_encoder.name += f'-epochs-{cond_encoder.epochs}'
    # Схема негативов входит в имя: иначе три схемы обучения писали бы
    # классификатор в один и тот же файл и затирали друг друга
    cond_encoder.augmentation_scheme = args.augmentation_scheme
    cond_encoder.name += f'-{cond_encoder.augmentation_scheme}'
    cond_encoder.name += artifact_suffix(config)
    # путь строится после того, как имя собрано целиком
    cond_encoder.cond_encoder_path = f"{data.base_path}/{data.datasets.datasets_list[0]}/{cond_encoder.name}.pth"
    # классификатор нужен только в режиме classifier guidance
    cond_encoder.use_conditional_encoder = config.classifier_guidance

    config.se_config = create_se_config()
    config.se_config.is_conditional = config.is_conditional
    # cross-attention нужен ровно тогда, когда условна сама диффузия.
    # В режиме classifier_guidance диффузия безусловная, cross-attention выключен.
    config.se_config.is_decoder = config.is_conditional
    config.se_config.vocab_size = AutoConfig.from_pretrained(model.encoder_link).vocab_size
    config.se_config.use_self_cond = config.use_self_cond

    # Метрики: в безусловном режиме оценивается качество текста как такового,
    # в условном и в classifier guidance -- соответствие промпту.
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

    config.project_name = args.project_name
    config.timesteps = "linear"
    pref = "emb" if config.emb else "actdm"
    training.checkpoints_prefix = f"{pref}-{model.encoder_name_hash}-{training.batch_size}-{optim.lr}-{data.datasets.datasets_list[0]}-cfg={data.swap_cfg_coef}"
    training.checkpoints_prefix += checkpoints_prefix_suffix(config)
    config.eval = args.eval or False

    print(f"[CONFIG] generation_mode={config.generation_mode}")
    print(f"[CONFIG] is_conditional={config.is_conditional} "
          f"(cross-attention: {'ON' if config.se_config.is_decoder else 'OFF'})")
    print(f"[CONFIG] classifier_guidance={config.classifier_guidance}, scale={config.guidance_scale}")
    print(f"[CONFIG] normalize_encodings={config.normalize_encodings}, emb={config.emb}")
    print(f"[CONFIG] split_scheme={data.split_scheme}, augmentation_scheme={cond_encoder.augmentation_scheme}")

    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = True
    config.save_top_k = 2
    return config


def artifact_suffix(config):
    """Суффикс, общий для декодера, классификатора и чекпоинтов диффузии.

    Все они обязаны жить в одном пространстве латентов и на одной нарезке данных,
    поэтому несовпадающие варианты не должны попадать в один файл. Значения по
    умолчанию дают пустой суффикс -- старые имена остаются валидными.
    """
    suffix = ""
    if not config.normalize_encodings and not config.emb:
        suffix += "-unnorm"
    if config.data.split_scheme != SPLIT_SCHEMES[0]:
        suffix += f"-{config.data.split_scheme}"
    return suffix


def checkpoints_prefix_suffix(config):
    """Суффикс имени чекпоинта диффузии.

    Режимы 1 и 3 используют ОДНУ И ТУ ЖЕ безусловную диффузию (режим 3 -- это
    та же модель плюс классификатор на генерации), поэтому суффикса у них нет
    и чекпоинт переиспользуется. У условной диффузии архитектура другая
    (есть cross-attention), поэтому чекпоинты разделены.
    """
    return ("-conditional" if config.is_conditional else "") + artifact_suffix(config)


def create_se_config():
    se_config = AutoConfig.from_pretrained("bert-base-cased")

    se_config.attention_head_size = se_config.hidden_size / se_config.num_attention_heads
    se_config.is_conditional = False
    se_config.use_self_cond = True

    if not hasattr(se_config, 'layer_norm_eps'):
        se_config.layer_norm_eps = 1e-12
    if not hasattr(se_config, 'hidden_dropout_prob'):
        se_config.hidden_dropout_prob = 0.1
    if not hasattr(se_config, 'attention_probs_dropout_prob'):
        se_config.attention_probs_dropout_prob = 0.1
    if not hasattr(se_config, 'initializer_range'):
        se_config.initializer_range = 0.02

    return se_config


def create_datasets_config(args):
    config = ml_collections.ConfigDict()
    config.downstream_tasks = ["qqp", "xsum", "paradetox", "wiki_auto"]
    if args.dataset_name is None:
        config.datasets_list = ["rocstories"]
    else:
        config.datasets_list = [args.dataset_name]
    config.metrics = {
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

    # набор метрик для rocstories выставляется в create_config,
    # когда уже известен режим работы (см. config.is_pipeline_conditional)

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
    config.epochs = 3
    config.lr = 1e-5
    config.weight_decay = 0.001
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
