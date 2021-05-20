import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Optional, Any

class InertialTransformer(nn.Transformer):
    def __init__(self, d_model: int = 8, nhead: int = 2, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 32, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 mean_src: float = 0, std_src: float = 1, mean_tgt: float = 0, std_tgt: float = 1) -> None:

        super(InertialTransformer, self).__init__(d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout,
                 activation, custom_encoder, custom_decoder)

        # Saving parameters for input standardization
        self.mean_src = torch.from_numpy(mean_src)
        self.std_src = torch.from_numpy(std_src)
        self.mean_tgt = torch.from_numpy(mean_tgt)
        self.std_tgt = torch.from_numpy(std_tgt)
        # Creating torch type 0 and 1 for later usage
        self.my_zero = torch.tensor(0, dtype=torch.float32)
        self.my_one = torch.tensor(1, dtype=torch.float32)
        # Defining 0 value tolerance
        self.float_tolerance = torch.tensor(10e-24, dtype=torch.float32)

        # Dropout layers applied on positional encoded inputs
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)

        # Linear layer applied on output of decoder stack
        self.last_linear = nn.Linear(d_model, d_model)

    # Generates upper triangle '-inf' mask
    def generate_subsequent_mask(self, q_s, k_s):
        mask = torch.triu(torch.ones(q_s, k_s), 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    # Function for standardization
    def standardize(self, x: Tensor, mean: Tensor, std: Tensor):
        # 0 constant fields will remain 0
        std = (torch.where((std - self.my_zero) < self.float_tolerance, self.my_one, std)).to(x.device)
        mean = mean.to(x.device)
        return (x - mean)/std

    def forward(self, src: Tensor, tgt: Tensor, pos_encoding: Tensor, src_seq_len: int) -> Tensor:
        # The source (encoder input) sequence length is defined by function argument
        # The target (decoder input) sequence length can vary from 1 to source sequence length
        # Predictions are generated until the length of decoder output sequence reaches the source sequence length
        # The last element of predicted sequence is concatenated to the next target sequence

        # Standardizing encoder input (source) and decoder input (target)
        src = self.standardize(src, self.mean_src, self.std_src)
        self.tgt = self.standardize(tgt, self.mean_tgt, self.std_tgt)

        # Adding positional encoding for source
        src += pos_encoding
        # Adding positional encoding for target
        pos_encoding_part = pos_encoding[:, :tgt.shape[1], :].reshape(self.tgt.shape)
        self.tgt += pos_encoding_part

        self.tgt_out = self.tgt

        # Applying dropout layer on positional encoded source sequence
        src = self.src_dropout(src)
        # Transposing due to the input format requirement of torch transformer model
        src = torch.transpose(src, 0, 1)

        # Generating Look-Ahead mask for encoder side
        src_mask = self.generate_subsequent_mask(src_seq_len, src_seq_len).to(src.device)

        # Predicting until reaching the same sequence length on the decoder output as the source sequence length
        for tgt_seq_len in range(tgt.shape[1], src_seq_len+1):
            # Applying dropout layer on positional encoded target sequence
            self.tgt = self.tgt_dropout(self.tgt)
            # Transposing due to the input format requirement of torch transformer model
            self.tgt_out = torch.transpose(self.tgt, 0, 1)

            # Generating the properly sized Look-Ahead mask for decoder side and for transition between the 2 sides
            tgt_mask = self.generate_subsequent_mask(tgt_seq_len, tgt_seq_len).to(src.device)
            memory_mask = self.generate_subsequent_mask(tgt_seq_len, src_seq_len).to(src.device)

            # Calling the forward method of torch transformer model
            self.tgt_out = super(InertialTransformer, self).forward(src, self.tgt_out, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)

            # Applying a last linear layer on decoder output
            self.tgt_out = self.last_linear(self.tgt_out)
            # Transposing the predicted output back
            self.tgt_out = torch.transpose(self.tgt_out, 0, 1)

            # In case of a necessary following iteration, the last element of predicted output sequence is standardized, positional encoded and concatenated to the input target sequence
            if tgt_seq_len != src_seq_len:
                tgt_last = self.standardize(self.tgt_out[:, -1, :].reshape(self.tgt_out.shape[0], 1, self.tgt_out.shape[-1]), self.mean_tgt, self.std_tgt)
                tgt_last += pos_encoding[:, tgt_seq_len, :].reshape(tgt_last.shape)

                self.tgt = torch.cat((self.tgt, tgt_last), 1)

        return self.tgt_out





