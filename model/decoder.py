import torch
import torch.nn as nn
from copy import deepcopy
from .score_estimator import BertBlock


# Декодеры, обученные в исходном tencdm, это BertDecoder: там три слоя лежали
# внутри HF BertEncoder, то есть под именами bert.layer.{i}.*, а здесь тот же
# слой называется blocks.{i}.*. Веса переносимы один в один:
#   * BertBlock здесь -- это переписанный BertLayer с теми же подмодулями
#     attention / intermediate / output и тем же порядком операций;
#   * в безусловном режиме исходный BertDecoder вызывался как decoder(latent),
#     без attention_mask -- ровно так, как считает Decoder.forward ниже.
# Совпадает и остальное пространство: score_estimator и encoder в форке не
# менялись, так что чекпоинт диффузии грузится как есть.
#
# Так что достаточно переименовать префикс. Загрузка остаётся strict:
# переименование обязано дать ТОЧНОЕ совпадение ключей, иначе часть весов
# осталась бы случайной и декодер молча врал бы текст.
LEGACY_BLOCK_PREFIX = "bert.layer."


def remap_legacy_decoder_state(state):
    """state_dict старого BertDecoder -> имена текущего Decoder.

    Современный чекпоинт возвращается тем же объектом без копирования.
    """
    if not any(k.startswith(LEGACY_BLOCK_PREFIX) for k in state):
        return state
    remapped = {}
    for key, value in state.items():
        if key.startswith(LEGACY_BLOCK_PREFIX):
            key = "blocks." + key[len(LEGACY_BLOCK_PREFIX):]
        remapped[key] = value
    print(f"Decoder: чекпоинт старого BertDecoder, "
          f"{LEGACY_BLOCK_PREFIX}* переименованы в blocks.*")
    return remapped


class Decoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()

        self.num_hidden_layers = decoder_config.num_hidden_layers
        
        arch_config = deepcopy(diffusion_config)
        arch_config.is_conditional = decoder_config.is_conditional
        arch_config.is_decoder = decoder_config.is_conditional
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(0, self.num_hidden_layers)]
        )
        self.fc = nn.Linear(arch_config.hidden_size, arch_config.vocab_size)

    def forward(self, x, cond_x=None, cond_mask=None):
        extended_cond_mask = self.get_extended_attention_mask(cond_mask)
        for _, block in enumerate(self.blocks):
            x = block(
                hidden_states=x,
                attention_mask=None,
                encoder_hidden_states=cond_x,
                encoder_attention_mask=extended_cond_mask
            )
        x = self.fc(x)
        return x 

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask is None:
            return None
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
        return extended_attention_mask
