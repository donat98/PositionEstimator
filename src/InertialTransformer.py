import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Optional, Any

class InertialTransformer(nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 mean_src: float = 0, std_src: float = 1, mean_tgt: float = 0, std_tgt: float = 1) -> None:

        super(InertialTransformer, self).__init__(d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, custom_encoder, custom_decoder)

        self.mean_src = torch.from_numpy(mean_src)
        self.std_src = torch.from_numpy(std_src)
        self.mean_tgt = torch.from_numpy(mean_tgt)
        self.std_tgt = torch.from_numpy(std_tgt)
        self.my_zero = torch.tensor(0, dtype=torch.float32)
        self.my_one = torch.tensor(1, dtype=torch.float32)
        self.float_tolerance = torch.tensor(10e-24, dtype=torch.float32)

        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)

        self.last_linear = nn.Linear(d_model, d_model)

    def generate_subsequent_mask(self, q_s, k_s):
        mask = torch.triu(torch.ones(q_s, k_s), 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def standardize(self, x: Tensor, mean: Tensor, std: Tensor):
        # 0 constant fields will remain 0
        std = (torch.where((std - self.my_zero) < self.float_tolerance, self.my_one, std)).to(x.device)
        mean = mean.to(x.device)
        return (x - mean)/std

    def forward(self, src: Tensor, tgt: Tensor, src_seq_len: int, tgt_seq_len: int) -> Tensor:

        src = self.standardize(src, self.mean_src, self.std_src)
        tgt = self.standardize(tgt, self.mean_tgt, self.std_tgt)

        src = self.src_dropout(src)
        tgt = self.tgt_dropout(tgt)

        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)

        src_mask = self.generate_subsequent_mask(src_seq_len, src_seq_len).to(src.device)
        tgt_mask = self.generate_subsequent_mask(tgt_seq_len, tgt_seq_len).to(src.device)
        memory_mask = self.generate_subsequent_mask(tgt_seq_len, src_seq_len).to(src.device)

        tgt = super(InertialTransformer, self).forward(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)

        tgt = self.last_linear(tgt)
        tgt = torch.transpose(tgt, 0, 1)

        return tgt





