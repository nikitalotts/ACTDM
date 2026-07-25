## Генерация через cross-attention и через latent replacement

Здесь собраны два подхода к подаче условия в диффузию. Они отличаются только этим, поэтому реализованы в одном проекте и выбираются параметром `--architecture_type`:

- `genie` -- условие подается в denoising network на уровне архитектуры: промпт проходит через отдельный энкодер и затем поступает в каждый блок сети через слои cross-attention. Подход адаптирован из метода Genie.
- `diffuseq` -- условие задается не через архитектуру, а через сам процесс генерации: латенты, соответствующие промпту, фиксируются на каждом шаге обратного процесса, а зашумление и расшумление выполняются только на латентах продолжения. Подход адаптирован из метода DiffuSeq.

Во всех скриптах архитектура задается переменной окружения `ARCH_TYPE` (по умолчанию `genie`), например:
```
ARCH_TYPE=diffuseq bash train_diffusion.sh                    # genie
ARCH_TYPE=diffuseq bash train_diffusion.sh # diffuseq
```
Декодер и чекпоинты диффузии у двух архитектур разные: для `genie` декодер условный, для `diffuseq` -- безусловный, а к префиксу чекпоинтов `diffuseq` добавляется суффикс `-diffuseq`. Поэтому декодер нужно обучить отдельно под каждую архитектуру.

## Зависимости

Список используемых библиотек приведен в [requirements.txt](./requirements.txt). Создать и активировать окружение можно с помощью Miniconda:
- `conda create --name actdm_genie_diffuseq python=3.9`
- `conda activate actdm_genie_diffuseq`
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
bash train_decoder.sh                    # genie
ARCH_TYPE=diffuseq bash train_decoder.sh # diffuseq
```

## Обучение диффузионной модели

После подсчета статистик и обучения декодера запустите обучение условной диффузионной модели:
```
bash train_diffusion.sh                    # genie
ARCH_TYPE=diffuseq bash train_diffusion.sh # diffuseq
```

## Оценка модели

Для генерации продолжений и подсчета метрик используйте:
```
bash eval_diffusion.sh        # одиночный запуск генерации и подсчета метрик
bash eval_diffusion_stat.sh   # многократный запуск для усреднения метрик и оценки разброса
```
