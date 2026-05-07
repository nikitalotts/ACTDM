## Генерация через cross-attention

В этом подходе условие подается в denoising network на уровне архитектуры:  промпт проходит через отдельный энкодер и затем поступает в каждый блок сети через слои cross-attention. Подход адаптирован из метода Genie.

## Зависимости

Список используемых библиотек приведен в [requirements.txt](./requirements.txt). Создать и активировать окружение можно с помощью Miniconda:
- `conda create --name actdm_genie python=3.9`
- `conda activate actdm_genie`
- `conda install pip`
- `pip install -r requirements.txt`
- `python -m spacy download en`

## Загрузка датасета

Скачать датасет ROCStories можно следующей командой:
```
python -m data.load --dataset_name=rocstories
```

## Подсчет статистик

Для корректного обучения диффузионной модели необходимо предварительно посчитать статистики:
```
bash make_statistics.sh
```

## Обучение декодера

Необходимо обучить декодер, восстанавливающий токены по энкодингам:
```
bash train_decoder.sh
```

## Обучение диффузионной модели

После подсчета статистик и обучения декодера запустите обучение условной диффузионной модели с cross-attention:
```
bash train_diffusion.sh
```

## Оценка модели

Для генерации продолжений и подсчета метрик используйте:
```
bash eval_diffusion.sh        # одиночный запуск генерации и подсчета метрик
bash eval_diffusion_stat.sh   # многократный запуск для усреднения метрик и оценки разброса
```
