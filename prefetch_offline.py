"""Прогрев кэшей HuggingFace для офлайн-запуска на кластере.

Запускать там, где есть интернет (login-нода), под тем же conda-окружением и
тем же HF_HOME/кэшем, которые увидят compute-ноды:

    python prefetch_offline.py

Скачивает ВСЕ модели, токенизаторы и скрипты метрик, которые пайплайн трогает
от обучения до финальной оценки. Без этого eval упадет на первом же вызове
mauve/ppl/bert-score -- уже после дней обучения.

Датасет wikipedia сюда намеренно не входит: стадия

    ./run_wikipedia.sh data

сама скачивает и нарезает его в datasets/wikipedia, после чего сырой кэш
~/.cache/huggingface/datasets/wikimedia___wikipedia* можно удалить --
обучение читает только datasets/wikipedia.

Суммарный объем кэша моделей: ~12-13 GB (основное: gpt-neo-1.3B ~5GB для ppl,
gpt2-large ~3GB для mauve, deberta-xlarge-mnli ~3GB для bert-score).
"""
import os

STEPS = []


def step(desc):
    def wrap(fn):
        STEPS.append((desc, fn))
        return fn
    return wrap


@step("BERT: энкодер латентов, классификатор, конфиг denoising-сети")
def bert():
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    # в коде встречаются оба имени: model.encoder_link = 'google-bert/...',
    # а create_se_config берет AutoConfig 'bert-base-cased'. Кэш huggingface
    # ключуется по запрошенному имени, поэтому греем оба
    for name in ("google-bert/bert-base-cased", "bert-base-cased"):
        AutoTokenizer.from_pretrained(name)
        AutoConfig.from_pretrained(name)
    AutoModel.from_pretrained("google-bert/bert-base-cased")


@step("GPT2-medium: только токенизатор и конфиг (веса обучаются с нуля)")
def gpt2_medium():
    from transformers import GPT2Tokenizer, GPT2Config
    GPT2Tokenizer.from_pretrained("gpt2-medium")
    GPT2Config.from_pretrained("gpt2-medium")


@step("Токенизатор mBERT (метрика bleu)")
def mbert():
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


@step("gpt2-large, ~3 GB (фичи для mauve)")
def gpt2_large():
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("gpt2-large")
    AutoModel.from_pretrained("gpt2-large")


@step("EleutherAI/gpt-neo-1.3B, ~5 GB (метрика ppl)")
def gpt_neo():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")


@step("microsoft/deberta-xlarge-mnli, ~3 GB (метрика bert-score)")
def deberta():
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
    AutoModel.from_pretrained("microsoft/deberta-xlarge-mnli")


@step("Скрипты метрик evaluate: mauve, rouge, bertscore, perplexity")
def evaluate_modules():
    from evaluate import load
    load("mauve")
    load("rouge")
    load("bertscore", module_type="metric")
    load("perplexity", module_type="metric")


@step("spacy en_core_web_sm (токенизация в diversity-метриках)")
def spacy_model():
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        # ставится как pip-пакет в текущее окружение, не в HF-кэш
        from spacy.cli import download
        download("en_core_web_sm")
        spacy.load("en_core_web_sm")


def main():
    print("HF_HOME =", os.environ.get("HF_HOME", "~/.cache/huggingface (по умолчанию)"))
    failed = []
    for desc, fn in STEPS:
        print(f"\n=== {desc} ===", flush=True)
        try:
            fn()
            print("OK", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            failed.append(desc)

    print("\n" + "=" * 70)
    if failed:
        print("НЕ СКАЧАЛОСЬ (запусти скрипт повторно или разберись с окружением):")
        for d in failed:
            print("  -", d)
        raise SystemExit(1)
    print("Все кэши прогреты. Дальше: ./run_wikipedia.sh data")
    print("После стадии data сырой кэш датасета можно удалить:")
    print("  rm -rf ~/.cache/huggingface/datasets/wikimedia___wikipedia*")


if __name__ == "__main__":
    main()
