import ml_collections
import os


def create_config_gpt2(args):
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

    config.finetuning = False
    config.seed = 0
    config.ddp = True
    config.is_conditional = True
    config.emb = False
    config.use_self_cond = False

    config.project_name = args.project_name
    training.checkpoints_prefix = f"gpt2-medium-512-{optim.lr}-{data.datasets.datasets_list[0]}"
    config.eval = False

    config.tracked_dataset = data.datasets.datasets_list[0]
    config.tracked_metric = data.datasets.metrics[config.tracked_dataset]["tracked_metric"]
    config.higher_better = True
    config.save_top_k = 2

    print(f"[CONFIG] GPT2-medium from scratch, dataset={data.datasets.datasets_list[0]}")
    print(f"[CONFIG] max_context_len={data.max_context_len}, max_sequence_len={data.max_sequence_len}")
    print(f"[CONFIG] training_iters={training.training_iters}, batch_size={training.batch_size}, accum_batch_steps={training.accum_batch_steps}")

    return config


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
        "wikipedia": {"metrics": ["mauve", "div", "ppl"], "tracked_metric": "mauve"},
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
