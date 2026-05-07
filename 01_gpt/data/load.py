from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import defaultdict
from create_config import create_config
from itertools import chain
import argparse
from huggingface_hub import HfApi
import nltk
import os


def download_wikipedia(dataset_path):

    dt = load_dataset("bigscience-data/roots_en_wikipedia")
    dt = dt["train"]
    dt = dt.remove_columns("meta")

    def split(batch):
        result = []
        for text in batch["text"]:
            texts = text.split("\n\n")
            result.append(texts)
        result = list(chain(*result))
        return {"text": result}
    
    dt = dt.map(
        split,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
    )

    min_symbols = 600
    dt = dt.filter(lambda b: len(b["text"]) >= min_symbols, num_proc=30)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def split_into_sents(batch):
        result = []
        for text in batch["text"]:
            texts = tokenizer.tokenize(text)
            result.append(texts)
        result = list(chain(*result))
        return {"text": result}
    
    sent_dt = dt.map(
        split_into_sents,
        batched=True,
        num_proc=30,
        desc="Dataset split",
        batch_size=1000,
    )

    def join_sents(batch):
        result = []
        cur_text = ''
        for text in batch["text"]:
            if len(cur_text.split()) + len(text.split()) < 128 / 2:
                cur_text += ' ' + text
            else:
                result.append(cur_text)
                cur_text = text
        
        return {"text": result}

    joined_dt = sent_dt.map(
        join_sents,
        batched=True,
        num_proc=30,
        desc="Dataset join",
        batch_size=100000,
    )

    dt = joined_dt.train_test_split(test_size=0.002, seed=0)
    dt.save_to_disk(
        dataset_path,
        num_shards={'train': 20, 'test': 1}
    )


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
    

def download_rocstory(dataset_path):
    print('Downloading ROCStories')
    def preprocess(batch):
        sources, targets = [], []
        size = len(batch["storyid"])
        for i in range(size):
            src = " ".join([batch[f"sentence{k}"][i] for k in range(1, 4)]) 
            trg = " ".join([batch[f"sentence{k}"][i] for k in range(4, 6)]) 
            sources.append(src)
            targets.append(trg)
        return {"source": sources, "target": targets}

    dt = load_dataset("wza/roc_stories")
    dt = dt["train"]
    dt = dt.map(
        preprocess,
        batched=True,
        num_proc=30,
        desc="Loading...",
        remove_columns=dt.column_names,
    )

    validation_size = 3000
    test_size = 7000

    dt_temp = dt.train_test_split(test_size=validation_size + test_size, seed=0, shuffle=True)
    dt_val_test = dt_temp["test"].train_test_split(test_size=test_size, seed=0, shuffle=True)

    dt = DatasetDict({
        "train": dt_temp["train"],
        "validation": dt_val_test["train"],
        "test": dt_val_test["test"],
    })

    print(f"Train: {len(dt['train'])}, Validation: {len(dt['validation'])}, Test: {len(dt['test'])}")

    dt.save_to_disk(dataset_path)

    print(f"\n{'=' * 60}")
    print("EXAMPLES FROM TRAIN SET")
    print(f"{'=' * 60}\n")

    for i in range(min(10, len(dt['train']))):
        example = dt['train'][i]
        print(f"Example {i + 1}:")
        print(f"  SRC: {example['source']}")
        print(f"  TRG: {example['target']}")
        print()

    print(f"{'=' * 60}\n")


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

    args = parser.parse_args()

    if args.dataset_name == "rocstories":
        download_rocstory(args.dataset_path + args.dataset_name)

    if args.dataset_name == "wikipedia":
        download_wikipedia(args.dataset_path + args.dataset_name)
    if args.dataset_name == "qqp":
        download_qqp(args.dataset_path + args.dataset_name)
    if args.dataset_name == "xsum":
        download_xsum(args.dataset_path + args.dataset_name)
    if args.dataset_name == "wiki_auto":
        download_wiki_auto(args.dataset_path + args.dataset_name)
