## Авторегрессионный baseline на основе GPT-2-medium

В качестве базовой модели для сравнения с диффузионными подходами используется авторегрессионная модель архитектуры GPT-2-medium, обучаемая с нуля на датасете ROCStories. 

## Зависимости

Список используемых библиотек приведен в [requirements.txt](./requirements.txt). Создать и активировать окружение можно с помощью Miniconda:
- `conda create --name actdm_gpt python=3.9`
- `conda activate actdm_gpt`
- `conda install pip`
- `pip install -r requirements.txt`
- `python -m spacy download en`

## Загрузка датасета

Скачать датасет ROCStories можно следующей командой:
```
python -m data.load --dataset_name=rocstories
```

## Обучение модели

После установки зависимостей и загрузки датасета запустите обучение GPT-2-medium с помощью скрипта:
```
bash train_gpt2.sh
```

## Оценка модели

Для генерации продолжений и подсчета метрик используйте:
```
bash eval_gpt2.sh        # одиночный запуск генерации и подсчета метрик
bash eval_gpt2_stat.sh   # многократный запуск для усреднения метрик и оценки разброса
```
