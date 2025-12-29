import functools
from collections import defaultdict
from typing import Union

import numpy as np
import random

random.seed(42)
np.random.seed(42)


def batch_preprocessing(batch, dataset_name, split, config):
    if dataset_name == "qqp":
        if split == "train":
            p_swap = 0.5
        else:
            p_swap = 0.

        new_batch = {
            "text_src": [],
            "text_trg": [],
        }
        for src, trg in zip(batch["question1"], batch["question2"]):
            if np.random.rand() < p_swap:
                src, trg = trg, src
            new_batch["text_src"].append(src)
            new_batch["text_trg"].append(trg)
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]
        
    elif dataset_name == "xsum":
        new_batch = {
            "text_src": batch["document"],
            "text_trg": batch["summary"],
        }
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]
        
    elif dataset_name == "wiki_auto":
        new_batch = {
            "text_src": batch["source"],
            "text_trg": batch["target"],
            "references": batch["references"],
        }
        new_batch["text_src"] = [f"Task is {dataset_name}. Prompt: {src}" for src in new_batch["text_src"]]

    elif dataset_name == "rocstories":
        if config.is_conditional:
            new_batch = {
                "text_src": [],
                "text_trg": [],
            }
            for sentences in batch["sentences"]:
                assert len(sentences) == 5, sentences
                # n_cond: сколько предложений в src (1-4)
                # n_trg: сколько предложений в trg (1 до 5-n_cond)

                n_cond = 4
                n_trg = 1

                # # на обучении делим случайно
                # if not config.eval:
                #     n_cond = random.randint(1, 4)
                #     max_trg = 5 - n_cond
                #     n_trg = random.randint(1, max_trg)
                
                text_cond = " ".join(sentences[:n_cond])
                text_trg = " ".join(sentences[n_cond:n_cond + n_trg])
                
                new_batch["text_src"].append(text_cond)
                new_batch["text_trg"].append(text_trg)
        else:
            new_batch = {
                "text_trg": batch["target"],
            }
        return new_batch

    # elif dataset_name == "rocstories":
    #     if task == 'train_coniditonal_encoder':
    #         n_cond = 1
    #         new_batch = {
    #             "text_src": [],
    #             "text_trg": [],
    #         }

    #         for sentences in batch["sentences"]:
    #             assert len(sentences) == 5, sentences
    #             text_cond = " ".join(sentences[:n_cond])
    #             text_trg = " ".join(sentences[n_cond:])

    #             new_batch["text_src"].append(text_cond)
    #             new_batch["text_trg"].append(text_trg)
    #     else:
    #         new_batch = {
    #             "text_trg": batch["target"],
    #         }

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    # CFG preprocessing
    swap_cfg_coef = config.data.swap_cfg_coef
    if split == "train" and swap_cfg_coef:
        length = len(new_batch["text_src"])
        swaps = (np.random.rand(length) < swap_cfg_coef)
        new_batch["text_src"] = ["" if swaps[i] else src for i, src in enumerate(new_batch["text_src"])]
        
    return new_batch
