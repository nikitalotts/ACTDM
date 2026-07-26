from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from itertools import chain
import argparse
import nltk
import os
from tqdm import tqdm

from utils.schemes import (
    SPLIT_SCHEMES, SPLIT_SCHEME_HELP, check_split_scheme, default_split_scheme,
)


def download_wikipedia(dataset_path, min_tokens=128, num_texts=None,
                       validation_size=3000, test_size=7000, seed=0):
    """Скачивает стандартный английский дамп Википедии и режет его на абзацы.

    Берется wikimedia/wikipedia 20231101.en -- это тот датасет, который научрук
    назвал стандартным (в отличие от bigscience-data/roots_en_wikipedia,
    использованного в TEncDM).

    На диск кладутся сырые абзацы в колонке "text". Разбиение на промпт и
    продолжение здесь НЕ делается: оно происходит на лету в
    data/dataset_wiki.py по config.data.split_scheme, потому что зависит от
    токенизатора энкодера и от max_context_len / max_sequence_len.

    min_tokens -- нижняя граница длины абзаца в словах. Абзацы короче
    отбрасываются: на них 50/50 разбиение дало бы слишком короткое продолжение.
    num_texts -- ограничение на число абзацев (для пробных прогонов).
    """
    print("Loading wikimedia/wikipedia 20231101.en ...")
    print(f"  min_tokens={min_tokens}, num_texts={num_texts or 'все'}")

    dt = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    def to_paragraphs(batch):
        result = []
        for text in batch["text"]:
            result.extend(p.strip() for p in text.split("\n\n"))
        return {"text": result}

    dt = dt.map(
        to_paragraphs,
        batched=True,
        num_proc=30,
        desc="Splitting articles into paragraphs",
        batch_size=1000,
        remove_columns=dt.column_names,
    )

    # длина в словах -- дешевая оценка сверху для длины в токенах:
    # у BERT-токенизатора токенов всегда не меньше, чем слов
    dt = dt.filter(lambda b: len(b["text"].split()) >= min_tokens,
                   num_proc=30, desc="Filtering short paragraphs")

    print(f"Paragraphs after filtering: {len(dt)}")

    if num_texts is not None and num_texts < len(dt):
        dt = dt.shuffle(seed=seed).select(range(num_texts))
        print(f"Limited to {len(dt)} paragraphs")

    holdout = validation_size + test_size
    if len(dt) <= holdout:
        raise Exception(
            f"Абзацев ({len(dt)}) не хватает на valid+test ({holdout}). "
            f"Уменьшите --min_tokens или увеличьте --num_texts"
        )

    tmp = dt.train_test_split(test_size=holdout, seed=seed, shuffle=True)
    val_test = tmp["test"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    dt = DatasetDict({
        "train": tmp["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    print(f"Train: {len(dt['train'])}, validation: {len(dt['validation'])}, test: {len(dt['test'])}")
    print(f"Saving to {dataset_path} ...")
    dt.save_to_disk(dataset_path)
    print("Done. Разбиение на промпт/продолжение делается на лету по --split_scheme")



def download_qqp(dataset_path):
    dt = load_dataset("glue", "qqp")
    dt = dt.filter(lambda x: x["label"] == 1)
    dt = dt.remove_columns(["label", "idx"])
    dt = concatenate_datasets([dt["train"], dt["validation"]])

    dt = dt.train_test_split(test_size=0.2, seed=0)
    dt_train = dt["train"]
    dt = dt["test"].train_test_split(test_size=0.5, seed=0)

    dt = DatasetDict(
        {
            "train": dt_train,
            "validation": dt["train"],
            "test": dt["test"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_xsum(dataset_path):
    dt = load_dataset("EdinburghNLP/xsum")
    dt.save_to_disk(dataset_path)


def download_wiki_auto(dataset_path):
    dt = load_dataset("GEM/wiki_auto_asset_turk")
    dt = dt.remove_columns(["gem_id", "gem_parent_id"])

    dt = DatasetDict(
        {
            "train": dt["train"],
            "validation": dt["validation"],
            "test": dt["test_asset"],
        }
    )
    dt.save_to_disk(dataset_path)


def download_squad(dataset_path):
    def make_batch(batch):
        new_batch = {
            "source": [],
            "target": [],
        }

        for context, answer, target in zip(batch["context"], batch["answers"], batch["target"]):
            if answer["text"]:
                new_batch["source"].append(f"Context: {context}. Answer: {answer['text'][0]}.")
                new_batch["target"].append(target)
        return new_batch

    dt = load_dataset("GEM/squad_v2")
    dt = dt.map(
        make_batch,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
        remove_columns=dt["train"].column_names
    )
    dt.save_to_disk(dataset_path)


def split_story(sentences, split_scheme):
    """Разбивает историю из 5 предложений на пары (промпт, продолжение).

    Возвращает список пар -- у sliding их три, у остальных схем одна.
    Единственное место, где задана нарезка rocstories.
    """
    assert len(sentences) == 5, f"Expected 5 sentences, got {len(sentences)}"

    if split_scheme == "last_sentence":
        return [(" ".join(sentences[:4]), sentences[4])]
    if split_scheme == "half":
        return [(" ".join(sentences[:3]), " ".join(sentences[3:5]))]
    if split_scheme == "sliding":
        return [
            (" ".join(sentences[:n]), " ".join(sentences[n:n + 2]))
            for n in range(1, 4)
        ]
    raise Exception(f"Unknown split_scheme: {split_scheme}. Expected one of {SPLIT_SCHEMES}")


def download_rocstory(dataset_path, split_scheme=SPLIT_SCHEMES[0]):

    print(f"Loading rocstories from HuggingFace...")
    print(f"Split scheme: {split_scheme} -- {SPLIT_SCHEME_HELP[split_scheme]}")

    def preprocess_sentences(batch):
        text_src_list = []
        text_trg_list = []
        sentences_list = []
        text_full_list = []

        size = len(batch["storyid"])
        for i in range(size):
            sentences = [batch[f"sentence{k}"][i] for k in range(1, 6)]
            sentences_list.append(sentences)

            text_full = " ".join(sentences)
            text_full_list.append(text_full)

            text_src, text_trg = split_story(sentences, split_scheme)[0]
            text_src_list.append(text_src)
            text_trg_list.append(text_trg)

        return {
            "sentences": sentences_list,
            "text_src": text_src_list,
            "text_trg": text_trg_list,
            "text_full": text_full_list,
        }

    dt = load_dataset("wza/roc_stories", trust_remote_code=True)
    dt = dt["train"]

    print("Preprocessing sentences...")
    dt = dt.map(
        preprocess_sentences,
        batched=True,
        num_proc=30,
        desc="Extracting sentences",
        remove_columns=dt.column_names,
    )

    print("Splitting into train/test...")

    total_examples = len(dt)
    print(f"Total examples: {total_examples}")

    validation_size = 3000
    test_size = 7000
    train_size = total_examples - validation_size - test_size

    print(f"Target sizes: train={train_size}, validation={validation_size}, test={test_size}")

    dt_temp_split = dt.train_test_split(
        test_size=validation_size + test_size, 
        seed=0,
        shuffle=True
    )

    dt_val_test = dt_temp_split["test"].train_test_split(
        test_size=test_size, 
        seed=0,
        shuffle=True
    )

    dt = DatasetDict({
        "train": dt_temp_split["train"], 
        "validation": dt_val_test["train"], 
        "test": dt_val_test["test"] 
    })

    print(f"\nActual sizes:")
    print(f"Train: {len(dt['train'])}")
    print(f"Validation: {len(dt['validation'])}")
    print(f"Test: {len(dt['test'])}")

    if split_scheme == "last_sentence":
        # одна пара на историю, уже собрана в preprocess_sentences
        dt = dt.remove_columns("sentences")
    else:
        print(f"\nApplying split scheme '{split_scheme}'...")

        formatted_datasets = {}
        for split in ["train", "validation", "test"]:
            print(f"\nProcessing {split} split...")
            split_dt = dt[split]

            formatted_examples = []
            for example in tqdm(split_dt, desc=f"Formatting {split}"):
                text_full = example["text_full"]
                for text_src, text_trg in split_story(example["sentences"], split_scheme):
                    formatted_examples.append({
                        "text_src": text_src,
                        "text_trg": text_trg,
                        "text_full": text_full,
                    })

            formatted_datasets[split] = Dataset.from_list(formatted_examples)
            print(f"{split.capitalize()}: {len(split_dt)} stories -> {len(formatted_datasets[split])} pairs")

        dt = DatasetDict(formatted_datasets)

    print(f"\nSaving to {dataset_path}...")
    dt.save_to_disk(dataset_path)

    print(f"\n{'=' * 60}")
    print(f"Dataset saved successfully to: {dataset_path}")
    print(f"Train examples: {len(dt['train'])}")
    print(f"Validation examples: {len(dt['validation'])}")
    print(f"Test examples: {len(dt['test'])}")

    print(f"Split scheme: {split_scheme} -- {SPLIT_SCHEME_HELP[split_scheme]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset arguments")
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        choices=[
            "rocstories",
            "wikipedia",
            "qqp", "xsum", "wiki_auto",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset_path", type=str, default=''.join([os.getcwd(), '/datasets/']),
        required=False,
    )
    parser.add_argument(
        "--split_scheme", type=str, default=None, choices=SPLIT_SCHEMES,
        help="Схема разбиения истории на промпт и продолжение: "
             + ", ".join(f"{k} -- {v}" for k, v in SPLIT_SCHEME_HELP.items()),
    )
    # --- только для wikipedia ---------------------------------------------------
    # разбиение на промпт/продолжение там делается на лету загрузчиком,
    # поэтому здесь задается лишь то, какие абзацы попадут в датасет
    parser.add_argument("--min_tokens", type=int, default=128,
                        help="wikipedia: минимальная длина абзаца в словах")
    parser.add_argument("--num_texts", type=int, default=None,
                        help="wikipedia: ограничить число абзацев (для пробных прогонов)")
    # старые флаги оставлены как алиасы, чтобы не ломать существующие команды
    parser.add_argument("--conditional_generation_formatted", action="store_true",
                        help="DEPRECATED, эквивалент --split_scheme sliding")
    parser.add_argument("--conditional_formatted_full_length", action="store_true",
                        help="DEPRECATED, эквивалент --split_scheme half")

    args = parser.parse_args()

    split_scheme = args.split_scheme or default_split_scheme(args.dataset_name)
    if args.conditional_formatted_full_length:
        split_scheme = "half"
    elif args.conditional_generation_formatted:
        split_scheme = "sliding"

    check_split_scheme(args.dataset_name, split_scheme)

    if args.dataset_name == "rocstories":
        download_rocstory(
            args.dataset_path + args.dataset_name,
            split_scheme=split_scheme,
        )

    if args.dataset_name == "wikipedia":
        download_wikipedia(
            args.dataset_path + args.dataset_name,
            min_tokens=args.min_tokens,
            num_texts=args.num_texts,
        )
    if args.dataset_name == "qqp":
        download_qqp(args.dataset_path + args.dataset_name)
    if args.dataset_name == "xsum":
        download_xsum(args.dataset_path + args.dataset_name)
    if args.dataset_name == "wiki_auto":
        download_wiki_auto(args.dataset_path + args.dataset_name)
