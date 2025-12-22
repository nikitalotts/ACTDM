import types
from datasets import load_from_disk
from create_config import create_config

def make_args_for_create_config():
    args = types.SimpleNamespace()
    args.scheduler = "none"
    args.coef_d = 1.0
    args.encoder_name = "bert-base-cased"
    args.emb = False
    args.mode = "transformer"
    args.swap_cfg_coef = 0.0
    args.project_name = "smoke_test"
    args.dataset_name = "rocstories"
    return args

def print_dataset_info(dataset, split_name):
    print("=" * 50)
    print(f"SPLIT: {split_name.upper()}")
    print("=" * 50)
    print(dataset)
    print("\n")
    
    # print("КОЛОНКИ:")
    # print(dataset.column_names)
    # print("\n")

    import  re
    print("ПРИМЕРЫ (первые 3):")
    # for i in range(min(3, len(dataset))):
    from tqdm import tqdm
    c = 10
    for r in tqdm(dataset):
        print(f"\n--- Примеры ---")
        print('SRC: ', r['text_src'])
        print('TRG: ', r['text_trg'])
        # sentences = re.split(r'(?<=[.!?])\s+', dataset[i][key])
        # print(len(sentences), sentences)
        # assert len(sentences) == 5, sentences
        c -= 1
        if c == 0:
            raise ValueError(10)

    # import  re
    # print("ПРИМЕРЫ (первые 3):")
    # # for i in range(min(3, len(dataset))):
    # from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     print(f"\n--- Пример {i+1} ---")
    #     for key in dataset.column_names:
    #         print(f"{key}: {dataset[i][key]}")
    #         sentences = re.split(r'(?<=[.!?])\s+', dataset[i][key])
    #         print(len(sentences), sentences)
    #         assert len(sentences) == 5, sentences
    
    print(f"\nВсего примеров в {split_name}: {len(dataset)}")
    print("=" * 50)
    print("\n")

def main():
    # Создаем конфиг
    args = make_args_for_create_config()
    config = create_config(args)
    
    base_path = f"{config.data.base_path}/rocstories"
    
    # Загружаем train split
    # train_dataset = load_from_disk(f"{base_path}/train")
    # print_dataset_info(train_dataset, "train")

    from data.dataset import get_dataset_iter

    def get_datasets(config):
        train_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="train", task='train_coniditonal_encoder')
        test_dataset = get_dataset_iter(config, dataset_name=config.decoder.dataset, split="test", task='train_coniditonal_encoder')
        return train_dataset, test_dataset

    train_dataset, valid_dataset = get_datasets(config=config)
    train_dataset = next(train_dataset)
    print_dataset_info(train_dataset, "train")

    # train_dataset = next(get_dataset_iter(config, dataset_name=config.decoder.dataset, split="train", task='train_coniditonal_encoder'))
    # Загружаем test split
    # test_dataset = load_from_disk(f"{base_path}/test")
    # print_dataset_info(test_dataset, "test")



if __name__ == "__main__":
    main()