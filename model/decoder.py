import torch
import torch.nn as nn
from copy import deepcopy
from .score_estimator import BertBlock
from transformers import AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

def custom_get_extended_attention_mask(attention_mask, dtype):
    if attention_mask is None:
        return None
    extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
    return extended_attention_mask


class BertDecoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()
        self.mode = decoder_config.mode
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.vocab_size = diffusion_config.vocab_size
        if decoder_config.mode == 'transformer':
            config.num_hidden_layers = decoder_config.num_hidden_layers
            if decoder_config.is_conditional:
                config.add_cross_attention = True
                config.is_decoder = True

            model = AutoModel.from_config(config)
            self.bert = model.encoder
            self.fc = nn.Linear(config.hidden_size, config.vocab_size)

            self.net = lambda x, **kwargs: self.fc(self.bert(x, **kwargs).last_hidden_state)
            self.get_extended_attention_mask = model.get_extended_attention_mask
        elif decoder_config.mode == 'mlm':
            self.cls = BertOnlyMLMHead(config)
            self.net = lambda x, **kwargs: self.cls(x)
            self.get_extended_attention_mask = custom_get_extended_attention_mask

    def forward(self, x, **kwargs):
        # Явно преобразуем вход в float
        if x.dtype != torch.float:
            x = x.float()
        
        if kwargs.get('encoder_attention_mask', None) is not None:
            kwargs['encoder_attention_mask'] = self.get_extended_attention_mask(kwargs['encoder_attention_mask'],
                                                                           dtype=x.dtype)
        return self.net(x, **kwargs)

    def decode_to_logits(self, x):
        return self.forward(x)


class Decoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()

        self.num_hidden_layers = decoder_config.num_hidden_layers
        
        arch_config = deepcopy(diffusion_config)
        arch_config.is_conditional = decoder_config.is_conditional
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(0, self.num_hidden_layers)]
        )
        self.fc = nn.Linear(arch_config.hidden_size, arch_config.vocab_size)

    def forward(self, x, cond_x=None, cond_mask=None):
        # Явно преобразуем вход в float
        if x.dtype != torch.float:
            x = x.float()
            
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