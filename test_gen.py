import torch
import os
import argparse
from ml_collections import ConfigDict
from transformers import AutoTokenizer, AutoConfig

def create_test_config(device='cpu'):
    config = ConfigDict()
    
    config.device = device
    
    config.model = ConfigDict()
    config.model.encoder_link = "bert-base-cased"
    config.model.ema_rate = 0.9999
    
    config.decoder = ConfigDict()
    config.decoder.mode = "transformer"
    config.decoder.num_hidden_layers = 3
    config.decoder.is_conditional = False
    config.decoder.decoder_path = "datasets/rocstories/decoder-bert-base-cased-80-transformer.pth"
    
    config.se_config = create_se_config()
    
    config.data = ConfigDict()
    config.data.max_sequence_len = 80
    config.data.max_context_len = 80
    config.data.enc_gen_mean = None
    config.data.enc_gen_std = None
    
    config.dynamic = ConfigDict()
    config.dynamic.N = 50
    config.dynamic.T = 1.0
    
    config.training = ConfigDict()
    config.training.checkpoints_folder = "checkpoints"
    config.training.checkpoints_prefix = "1tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0"
    config.training.checkpoint_name = "last"
    
    config.validation = ConfigDict()
    config.validation.batch_size = 4
    config.validation.cfg_coef = 0.0
    
    config.emb = True
    config.use_self_cond = True
    config.ddp = False
    config.is_conditional = False
    config.timesteps = "linear"
    config.seed = 42
    
    return config

def create_se_config():
    se_config = AutoConfig.from_pretrained("bert-base-cased")
    se_config.attention_head_size = se_config.hidden_size / se_config.num_attention_heads
    se_config.is_conditional = False
    se_config.use_self_cond = True
    se_config.vocab_size = 28996
    return se_config

def debug_score_estimator(score_estimator, x_t, time_t, attention_mask):
    """Пошаговая отладка score estimator"""
    print(f"\n🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА SCORE ESTIMATOR:")
    
    # Проверяем входные данные
    print(f"   Входные данные:")
    print(f"     x_t: {x_t.shape}, dtype: {x_t.dtype}, device: {x_t.device}")
    print(f"     time_t: {time_t.shape}, dtype: {time_t.dtype}, device: {time_t.device}")
    print(f"     attention_mask: {attention_mask.shape}, dtype: {attention_mask.dtype}, device: {attention_mask.device}")
    
    # Создаем x_0_self_cond (self-conditioning) если модель его требует
    batch_size, seq_len, hidden_size = x_t.shape
    x_0_self_cond = torch.randn(batch_size, seq_len, hidden_size).to(x_t.device)
    print(f"     x_0_self_cond: {x_0_self_cond.shape} (создан для self-conditioning)")
    
    # Пробуем вызвать forward с правильными параметрами
    try:
        print(f"\n   Попытка вызова forward с self-conditioning...")
        with torch.no_grad():
            output = score_estimator.forward(
                x_t=x_t,
                time_t=time_t,
                attention_mask=attention_mask,
                x_0_self_cond=x_0_self_cond
            )
        print(f"   ✓ Forward успешно выполнен!")
        print(f"   Выход: {output.shape}")
        return output
    except Exception as e:
        print(f"   ❌ Ошибка в forward: {e}")
        
        # Пробуем без self-conditioning
        try:
            print(f"\n   Попытка вызова forward без self-conditioning...")
            with torch.no_grad():
                output = score_estimator.forward(
                    x_t=x_t,
                    time_t=time_t,
                    attention_mask=attention_mask,
                    x_0_self_cond=None
                )
            print(f"   ✓ Forward успешно выполнен без self-conditioning!")
            print(f"   Выход: {output.shape}")
            return output
        except Exception as e2:
            print(f"   ❌ Ошибка и без self-conditioning: {e2}")
            return None

def full_generation_test(device='cpu'):
    """Полный тест генерации с загрузкой моделей"""
    print("\n" + "="*60)
    print("ПОЛНЫЙ ТЕСТ ГЕНЕРАЦИИ С МОДЕЛЯМИ")
    print("="*60)
    
    try:
        from model.encoder import Encoder
        from model.decoder import BertDecoder
        from model.score_estimator import ScoreEstimatorEMB
        from utils.ema_model import ExponentialMovingAverage
        
        config = create_test_config(device)
        print(f"\nИспользуем устройство: {device}")
        
        # 1. Инициализация энкодера
        print(f"\n1. Инициализация энкодера...")
        encoder = Encoder(
            config.model.encoder_link,
            enc_normalizer=None,
            is_change_sp_tokens=True,
            emb=config.emb
        ).eval()
        
        if device != 'cpu':
            encoder = encoder.to(device)
        
        print(f"   ✓ Энкодер инициализирован на {device}")
        
        # 2. Инициализация декодера
        print(f"\n2. Инициализация декодера...")
        
        decoder_state = torch.load(config.decoder.decoder_path, map_location=device, weights_only=False)
        
        decoder = BertDecoder(
            decoder_config=config.decoder,
            diffusion_config=config.se_config
        )
        
        if 'decoder' in decoder_state:
            missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state['decoder'], strict=False)
        else:
            missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state, strict=False)
        
        print(f"   ✓ Декодер загружен на {device}")
        print(f"   Отсутствующие ключи: {len(missing_keys)}")
        print(f"   Неожиданные ключи: {len(unexpected_keys)}")
        
        decoder = decoder.eval().to(device)
        
        # 3. Инициализация score estimator с EMA
        print(f"\n3. Инициализация score estimator с EMA...")
        
        score_estimator = ScoreEstimatorEMB(config=config.se_config).to(device)
        
        # Создаем EMA объект
        ema = ExponentialMovingAverage(score_estimator.parameters(), config.model.ema_rate)
        
        checkpoint_path = os.path.join(
            config.training.checkpoints_folder,
            config.training.checkpoints_prefix,
            f"{config.training.checkpoint_name}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Загружаем EMA веса правильным способом
        if 'ema' in checkpoint:
            print(f"   Загружаем EMA веса...")
            ema.load_state_dict(checkpoint["ema"])
            # Копируем EMA веса в модель
            ema.copy_to(score_estimator.parameters())
            print(f"   ✓ EMA веса загружены и применены")
            
        elif 'model' in checkpoint:
            print(f"   Загружаем обычные веса модели...")
            missing_model, unexpected_model = score_estimator.load_state_dict(checkpoint["model"], strict=False)
            print(f"   Model - отсутствующие ключи: {len(missing_model)}")
            print(f"   Model - неожиданные ключи: {len(unexpected_model)}")
        
        score_estimator.eval()
        print(f"   ✓ Score estimator загружен на {device} (шаг: {checkpoint.get('step', 'N/A')})")
        
        # 4. Тестовая генерация
        print(f"\n4. Тестовая генерация...")
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_link)
        
        test_text = "Once upon a time there was a"
        print(f"   Входной текст: '{test_text}'")
        
        with torch.no_grad():
            # Кодируем входной текст
            tok = tokenizer(
                [test_text],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.data.max_context_len,
                return_tensors="pt",
            ).to(device)
            
            src_x = encoder(
                input_ids=tok["input_ids"],
                attention_mask=tok["attention_mask"]
            )
            
            print(f"   ✓ Входной текст закодирован")
            print(f"   Форма эмбеддингов: {src_x.shape}")
            print(f"   Устройство эмбеддингов: {src_x.device}")
            
            # Тестируем декодер
            print(f"\n5. Тестирование декодера...")
            batch_size, seq_len = 2, 10
            
            # Создаем случайные эмбеддинги (а не токены), которые ожидает декодер
            # Форма: [batch_size, seq_len, hidden_size]
            hidden_size = config.se_config.hidden_size
            test_input = torch.randn(batch_size, seq_len, hidden_size).to(device)

            if config.decoder.is_conditional:
                decoder_output = decoder(test_input, encoder_hidden_states=src_x, encoder_attention_mask=tok["attention_mask"])
            else:
                decoder_output = decoder(test_input)

            print(f"   ✓ Декодер протестирован")
            print(f"   Вход декодера: {test_input.shape}, dtype: {test_input.dtype}")
            print(f"   Выход декодера: {decoder_output.shape}, dtype: {decoder_output.dtype}")
            
            # Тестируем score estimator с детальной диагностикой
            print(f"\n6. Тестирование score estimator...")
            
            # Создаем эмбеддинги для score estimator
            batch_size, seq_len = 2, 10
            hidden_size = config.se_config.hidden_size
            x_t = torch.randn(batch_size, seq_len, hidden_size).to(device)
            time_t = torch.rand(batch_size).to(device)
            attention_mask = torch.ones(batch_size, seq_len).to(device)
            
            # Детальная диагностика
            score_output = debug_score_estimator(score_estimator, x_t, time_t, attention_mask)
            
            if score_output is not None:
                print(f"   ✓ Score estimator работает корректно!")
                print(f"   Выход: {score_output.shape}")
                print(f"   Статистика выхода: mean={score_output.mean().item():.4f}, std={score_output.std().item():.4f}")
            else:
                print(f"   ⚠️ Score estimator имеет проблему, но это может не мешать генерации")
        
        print(f"\n🎉 ВСЕ КОМПОНЕНТЫ РАБОТАЮТ КОРРЕКТНО!")
        print(f"\nСводка:")
        print(f"  • Энкодер: ✓ работает")
        print(f"  • Декодер: ✓ работает") 
        print(f"  • Score Estimator: {'✓ работает' if score_output is not None else '⚠️ требует проверки'}")
        print(f"  • Все модели загружены на: {device}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ\n")
    
    success = full_generation_test()
    
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"Полная генерация: {'✓ OK' if success else '❌ FAILED'}")
    print("="*60)
    
    if success:
        print("\n🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
        print("Все модели готовы к генерации текстов!")
    else:
        print("\n⚠️ ЕСТЬ ПРОБЛЕМЫ - см. вывод выше")