"""
Скрипт для тестовой генерации текстов моделью диффузии.
Переиспользует код из test_gen.py и diffusion_holder.py
"""

import os
import json
import torch
import argparse
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from transformers import AutoTokenizer, AutoConfig

# Импорты из проекта (как в test_gen.py и diffusion_holder.py)
from model.encoder import Encoder
from model.decoder import BertDecoder
from model.score_estimator import ScoreEstimatorEMB
from utils.ema_model import ExponentialMovingAverage
from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver


def create_se_config(encoder_name="bert-base-cased", is_conditional=False, use_self_cond=True):
    """Создание конфига для score estimator (из test_gen.py)"""
    se_config = AutoConfig.from_pretrained(encoder_name)
    se_config.attention_head_size = se_config.hidden_size / se_config.num_attention_heads
    se_config.is_conditional = is_conditional
    se_config.use_self_cond = use_self_cond
    se_config.vocab_size = 28996  # bert-base-cased
    return se_config


def create_config(args):
    """Создание конфига (адаптировано из test_gen.py)"""
    from ml_collections import ConfigDict

    config = ConfigDict()
    config.device = args.device

    # Model
    config.model = ConfigDict()
    config.model.encoder_link = args.encoder_name
    config.model.ema_rate = 0.9999

    # Decoder
    config.decoder = ConfigDict()
    config.decoder.mode = args.decoder_mode
    config.decoder.num_hidden_layers = args.decoder_layers
    config.decoder.is_conditional = args.is_conditional
    config.decoder.decoder_path = args.decoder_path

    # Score estimator config
    config.se_config = create_se_config(
        encoder_name=args.encoder_name,
        is_conditional=args.is_conditional,
        use_self_cond=args.use_self_cond
    )

    # Data
    config.data = ConfigDict()
    config.data.max_sequence_len = args.max_seq_len
    config.data.max_context_len = args.max_ctx_len
    config.data.enc_gen_mean = None
    config.data.enc_gen_std = None

    # Dynamic
    config.dynamic = ConfigDict()
    config.dynamic.N = args.num_steps
    config.dynamic.T = 1.0
    config.dynamic.scheduler = "sd"
    config.dynamic.coef_d = 1.0
    config.dynamic.solver = "euler"

    # Training/checkpoints
    config.training = ConfigDict()
    config.training.checkpoints_folder = args.checkpoints_folder
    config.training.checkpoints_prefix = args.checkpoints_prefix
    config.training.checkpoint_name = args.checkpoint_name
    config.training.ode_sampling = args.ode_sampling

    # Validation
    config.validation = ConfigDict()
    config.validation.batch_size = args.batch_size
    config.validation.cfg_coef = 0.0

    # General
    config.emb = args.emb
    config.use_self_cond = args.use_self_cond
    config.is_conditional = args.is_conditional
    config.timesteps = args.timesteps
    config.seed = args.seed
    config.solver = "euler"

    return config


class TextGenerator:
    """Генератор текстов (логика из diffusion_holder.py)"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        print(f"Инициализация на устройстве: {self.device}")

        # Tokenizer
        print("1. Загрузка токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_link)

        # Encoder (как в test_gen.py)
        print("2. Загрузка энкодера...")
        self.encoder = Encoder(
            config.model.encoder_link,
            enc_normalizer=None,
            is_change_sp_tokens=True,
            emb=config.emb
        ).eval().to(self.device)

        # Decoder (как в test_gen.py)
        print("3. Загрузка декодера...")
        self.decoder = BertDecoder(
            decoder_config=config.decoder,
            diffusion_config=config.se_config
        )
        self._load_decoder()
        self.decoder = self.decoder.eval().to(self.device)

        # Score estimator (как в test_gen.py)
        print("4. Загрузка score estimator...")
        se_config = deepcopy(config.se_config)
        se_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(config=se_config).to(self.device)

        # EMA
        self.ema = ExponentialMovingAverage(
            self.score_estimator.parameters(),
            config.model.ema_rate
        )
        self._load_checkpoint()
        self.score_estimator.eval()

        # Dynamic и solver (как в diffusion_holder.py)
        print("5. Инициализация диффузии...")
        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.score_estimator),
            ode_sampling=config.training.ode_sampling
        )

        print("Генератор готов!\n")

    def _load_decoder(self):
        """Загрузка декодера (как в diffusion_holder.py restore_decoder)"""
        decoder_path = self.config.decoder.decoder_path
        if os.path.exists(decoder_path):
            state = torch.load(decoder_path, map_location=self.device, weights_only=False)
            if 'decoder' in state:
                self.decoder.load_state_dict(state['decoder'], strict=False)
            else:
                self.decoder.load_state_dict(state, strict=False)
            print(f"   Декодер загружен: {decoder_path}")
        else:
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    def _load_checkpoint(self):
        """Загрузка чекпоинта с EMA (как в test_gen.py)"""
        prefix_folder = os.path.join(
            self.config.training.checkpoints_folder,
            self.config.training.checkpoints_prefix
        )

        name = self.config.training.checkpoint_name
        if not name:
            checkpoint_names = os.listdir(prefix_folder)
            checkpoint_names = [t.replace(".pth", "") for t in checkpoint_names]
            checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]
            name = str(max(checkpoint_names)) if checkpoint_names else "last"

        checkpoint_path = os.path.join(prefix_folder, f"{name}.pth")
        print(f"   Загрузка: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
            self.ema.copy_to(self.score_estimator.parameters())
            print(f"   EMA веса применены (step: {checkpoint.get('step', 'N/A')})")
        elif 'model' in checkpoint:
            self.score_estimator.load_state_dict(checkpoint['model'], strict=False)
            print(f"   Model веса загружены (step: {checkpoint.get('step', 'N/A')})")

    def calc_score(self, model, x_t, t, cond=None, attention_mask=None,
                   cond_mask=None, x_0_self_cond=None):
        """Вычисление score (из diffusion_holder.py)"""
        params = self.dynamic.marginal_params(t)
        x_0 = model(
            x_t=x_t, time_t=t, cond=cond,
            attention_mask=attention_mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        )
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {"score": score, "x_0": x_0, "eps_theta": eps_theta}

    @torch.no_grad()
    def pred_embeddings(self, batch_size, cond_x=None, cond_mask=None, attention_mask=None):
        """Генерация эмбеддингов (из diffusion_holder.py)"""
        self.score_estimator.eval()

        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder.encoder.config.hidden_size
        )

        # Начинаем с шума
        x = self.dynamic.prior_sampling(shape).to(self.device)
        x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
        eps_t = 0.01

        # Временные шаги
        if self.config.timesteps == "linear":
            timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)
        elif self.config.timesteps == "quad":
            deg = 2
            timesteps = torch.linspace(1, 0, self.dynamic.N + 1, device=self.device) ** deg * (self.dynamic.T - eps_t) + eps_t
        else:
            timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)

        # Обратный процесс
        for idx in tqdm(range(self.dynamic.N), desc="Diffusion"):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]

            input_t = t * torch.ones(shape[0], device=self.device)
            next_input_t = next_t * torch.ones(shape[0], device=self.device)

            output = self.diff_eq_solver.step(
                x_t=x, t=input_t, next_t=next_input_t,
                cond=cond_x, cond_mask=cond_mask,
                attention_mask=attention_mask,
                x_0_self_cond=x_0_self_cond,
            )

            x, x_mean = output["x"], output["x_mean"]
            x_0_self_cond = output["x_0"]

        return x_mean

    @torch.no_grad()
    def pred_logits(self, pred_embeddings, cond_x=None, cond_mask=None):
        """Предсказание логитов (из diffusion_holder.py)"""
        # Когда emb=True, cond_x и cond_mask должны быть None (как в diffusion_holder.py)
        if self.config.emb:
            cond_x = None
            cond_mask = None

        # BertDecoder с mode='transformer' ожидает encoder_hidden_states/encoder_attention_mask
        if self.config.decoder.mode == 'transformer':
            if cond_x is not None:
                output = self.decoder(
                    pred_embeddings,
                    encoder_hidden_states=cond_x,
                    encoder_attention_mask=cond_mask
                )
            else:
                # Без conditional - просто передаём эмбеддинги
                output = self.decoder(pred_embeddings)
        else:
            # Для mlm режима
            output = self.decoder(pred_embeddings)
        return output

    @torch.no_grad()
    def generate_text_batch(self, batch_size, cond_x=None, attention_mask=None, cond_mask=None):
        """Генерация батча текстов (из diffusion_holder.py)"""
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        pred_embeddings = self.pred_embeddings(
            batch_size=batch_size,
            attention_mask=attention_mask,
            cond_x=cond_x,
            cond_mask=cond_mask,
        )

        output = self.pred_logits(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
        tokens = output.argmax(dim=-1)

        # Найти конец последовательности
        end_tokens = []
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.eos_token])
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.sep_token])

        tokens = tokens.detach().cpu().tolist()
        tokens_list = []
        for seq in tokens:
            idx = 0
            while idx < len(seq) and seq[idx] not in end_tokens:
                idx += 1
            tokens_list.append(seq[0:idx])

        text = self.tokenizer.batch_decode(tokens_list, skip_special_tokens=True)
        return text, pred_embeddings

    @torch.no_grad()
    def generate(self, num_samples, batch_size=None):
        """Генерация заданного количества текстов"""
        if batch_size is None:
            batch_size = self.config.validation.batch_size

        all_texts = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_texts))

            print(f"\nБатч {i+1}/{num_batches} (size={current_batch_size})")
            texts, _ = self.generate_text_batch(batch_size=current_batch_size)
            all_texts.extend(texts)

            print(f"Сгенерировано: {len(all_texts)}/{num_samples}")

        return all_texts[:num_samples]


def parse_args():
    parser = argparse.ArgumentParser(description='Генерация текстов моделью диффузии')

    # Основные
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)

    # Модель
    parser.add_argument('--encoder-name', type=str, default='bert-base-cased')
    parser.add_argument('--decoder-mode', type=str, default='transformer')
    parser.add_argument('--decoder-layers', type=int, default=3)
    parser.add_argument('--decoder-path', type=str,
                        default='datasets/rocstories/decoder-bert-base-cased-80-transformer.pth')

    # Чекпоинты
    parser.add_argument('--checkpoints-folder', type=str, default='checkpoints')
    parser.add_argument('--checkpoints-prefix', type=str,
                        default='1tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0')
    parser.add_argument('--checkpoint-name', type=str, default='last')

    # Диффузия
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--max-seq-len', type=int, default=80)
    parser.add_argument('--max-ctx-len', type=int, default=80)
    parser.add_argument('--timesteps', type=str, default='linear', choices=['linear', 'quad'])
    parser.add_argument('--ode-sampling', action='store_true', default=False)

    # Флаги
    parser.add_argument('--emb', action='store_true', default=True)
    parser.add_argument('--use-self-cond', action='store_true', default=True)
    parser.add_argument('--is-conditional', action='store_true', default=False)

    # Вывод
    parser.add_argument('--output-file', type=str, default='generated_texts.json')

    return parser.parse_args()


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Конфиг
    config = create_config(args)

    # Генератор
    generator = TextGenerator(config)

    # Генерация
    print(f"\n{'='*50}")
    print(f"Генерация {args.num_samples} текстов...")
    print(f"{'='*50}\n")

    texts = generator.generate(num_samples=args.num_samples, batch_size=args.batch_size)

    # Вывод примеров
    print(f"\n{'='*50}")
    print("СГЕНЕРИРОВАННЫЕ ТЕКСТЫ:")
    print(f"{'='*50}")
    for i, text in enumerate(texts):
        print(f"\n[{i+1}] {text}")

    # Сохранение
    results = {
        "config": {
            "encoder": args.encoder_name,
            "checkpoint": args.checkpoints_prefix,
            "num_steps": args.num_steps,
            "num_samples": len(texts),
        },
        "texts": texts
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Результаты сохранены в: {args.output_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()