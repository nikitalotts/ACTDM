import os
import gc
import torch
import numpy as np
from random import random
import torch.distributed as dist
from typing import List
import datasets
from datasets import Dataset, load_from_disk
from itertools import cycle
from transformers import AutoTokenizer


class WikipediaDatasetDDP:
    def __init__(self, config, dataset_name, split):
        self.split = split
        self.config = config
        self.dataset_name = dataset_name
        self.base_path = f"{config.data.base_path}/{dataset_name}"
        self.device_id = dist.get_rank() if torch.distributed.is_initialized() else 0
        self.total_device_number = dist.get_world_size() if torch.distributed.is_initialized() else 1
        self.epoch = 0
        self.files = self.get_files()
        self.iter_files = cycle(self.files)
        self.max_context_len = config.data.max_context_len
        self.max_sequence_len = config.data.max_sequence_len
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_link)

    def get_files(self):
        path = f"{self.base_path}/{self.split}/"
        files = list(os.listdir(path))
        files = [t for t in files if ".arrow" in t]
        files = sorted(files, key = lambda f: int(f.split("-")[1]))
        return files

    def spilt_data_across_gpu(self, dt: List[str]):
        self.epoch += 1
        if self.split == "train":
            indexes = np.random.default_rng(seed=self.epoch).permutation(len(dt))
        else:
            indexes = np.arange(len(dt))
        
        start_ind = self.device_id * (len(dt) // self.total_device_number)
        end_ind = (self.device_id + 1) * (len(dt) // self.total_device_number)
        if (self.device_id + 1) == self.total_device_number:
            indexes = indexes[start_ind:]
        else:
            indexes = indexes[start_ind: end_ind]
        
        return Dataset.from_dict(dt[indexes])

    def load_data(self):
        file = next(self.iter_files)
        path = f"{self.base_path}/{self.split}/{file}"
        dt = Dataset.from_file(path)
        dt = self.spilt_data_across_gpu(dt)
        
        self.dt = dt.map(
            self.batch_preprocessing,
            batched=True,
            load_from_cache_file=False,
            num_proc=50,
            desc="Dataset preprocessing",
            batch_size=1000,
        )
        return self.dt
    
    def batch_preprocessing(self, batch):
        # промпт нужен и условной диффузии, и классификатору в режиме guidance,
        # и авторегрессионному бейзлайну -- то есть всем режимам кроме unconditional
        if self.config.is_pipeline_conditional:
            return self.batch_preprocessing_cond(batch)
        else:
            return self.batch_preprocessing_uncond(batch)

    def batch_preprocessing_uncond(self, batch): 
        if "text" in batch:
            return {"text_trg": batch["text"]}
        elif "target" in batch:
            return {"text_trg": batch["target"]}

    def batch_preprocessing_cond(self, batch):
        """Режет текст на промпт и продолжение по токенам.

        prefix_lm (схема prefix LM из TESS-2, по умолчанию):
            текст обрезается до max_context_len + max_sequence_len токенов
            и делится ровно пополам. Граница всегда на одной и той же позиции,
            поэтому позиционная статистика таргета не размазывается, а у diffuseq
            стык промпта и продолжения в конкатенированной последовательности
            стоит на фиксированном месте.

        random_prefix (прежнее поведение):
            длина промпта случайна в пределах max_context_len. Оставлено, чтобы
            можно было воспроизвести сравнение схем.
        """
        scheme = self.config.data.split_scheme
        total_len = self.max_context_len + self.max_sequence_len

        if self.split == 'train':
            blank_cond_rate = self.config.data.swap_cfg_coef
        else:
            blank_cond_rate = 0

        batch_input_ids = self.tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        batch_size = len(batch_input_ids)

        if scheme == "prefix_lm":
            # ровно 50% доступных токенов в промпт: для текстов длиннее total_len
            # это ровно max_context_len, для более коротких -- их собственная середина
            delimeter_poses = [min(len(ids), total_len) // 2 for ids in batch_input_ids]
        elif scheme == "random_prefix":
            delimeter_poses = (np.random.rand(batch_size) * (self.max_context_len - 1)).astype(int)
        else:
            raise Exception(
                f"split_scheme={scheme} не применим к wikipedia. "
                f"Ожидается prefix_lm или random_prefix"
            )

        trg_ids_list = []
        src_ids_list = []

        for i, input_ids in enumerate(batch_input_ids):
            if random() < blank_cond_rate:
                delimeter_poses[i] = 0
            pos = delimeter_poses[i]
            src_ids = input_ids[:pos]
            trg_ids = input_ids[pos:self.max_sequence_len + pos]
            if not trg_ids:
                src_ids, trg_ids = trg_ids, src_ids
            src_ids_list.append(src_ids)
            trg_ids_list.append(trg_ids)

        texts_src = self.tokenizer.batch_decode(src_ids_list)
        texts_trg = self.tokenizer.batch_decode(trg_ids_list)

        output = {
            "text_src": texts_src,
            "text_trg": texts_trg,
        }
        return output

    def get_data(self):
        while True:
            yield self.load_data()
            del self.dt
            gc.collect()
