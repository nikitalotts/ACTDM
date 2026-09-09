import os
import gc
import torch
import numpy as np
from random import random
import torch.distributed as dist
from typing import List
import datasets
from datasets import Dataset, load_from_disk, concatenate_datasets, DatasetDict

from .dataset_wiki import WikipediaDatasetDDP
from .dataset_tasks import DownstreamTaskDatasetDDP


def get_dataset_iter(config, dataset_name, split):
    if dataset_name.startswith("wikipedia"):
        dt = WikipediaDatasetDDP(config, dataset_name, split)
    else:
        dt = DownstreamTaskDatasetDDP(config, dataset_name, split)
    return dt.get_data()


def take_fixed_subset(dt, num_examples, seed=0):
    """Детерминированная подвыборка фиксированного размера.

    Wikipedia отдается шардами примерно по 1.5 млн пар, а бюджет обучения
    классификатора покрывает лишь часть шарда. Без явного отбора эта «часть»
    определялась бы перемешиванием DataLoader-а: она менялась бы от запуска к
    запуску и различалась бы между тремя схемами классификатора. Три схемы
    обучались бы на разных данных, и разница между ними перестала бы быть
    разницей между схемами.

    Отбираем один и тот же пул по фиксированному сиду. Индексы возвращаем по
    возрастанию: набор от этого не меняется, а чтение arrow-файла остается
    последовательным. Перемешивание -- забота DataLoader-а.
    """
    if not num_examples:
        return dt
    if num_examples >= len(dt):
        if num_examples > len(dt):
            print(f"WARNING: запрошен пул из {num_examples} примеров, "
                  f"в шарде только {len(dt)} -- берем все", flush=True)
        return dt
    idx = np.random.default_rng(seed).permutation(len(dt))[:num_examples]
    return dt.select(sorted(int(i) for i in idx))


class DatasetDDP:
    def __init__(self, config, split):
        self.split = split
        self.config = config
        self.datasets_config = config.data.datasets
        
        self.datasets = dict()
        for dataset_name in self.datasets_config.datasets_list:
            self.datasets[dataset_name] = get_dataset_iter(self.config, dataset_name, self.split)

    def load_data(self):
        datasets = []
        for dataset_name, dt_iter in self.datasets.items():
            datasets.append(next(dt_iter)) 
        dt = concatenate_datasets(datasets)
        return dt

    def get_data(self):
        while True:
            yield self.load_data()
            
