# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:45:58 2023

@author: hwang147
"""

import torch
import torch.nn as nn


from model.Attention import embedding 
from model.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout) for _ in range(N)])
        

        self._embedding = embedding(d_input, d_model)
        self._linear2 = nn.Linear(d_model,2)
        self._linear = nn.Linear(2, d_output)
        self.name = 'transformer'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self._embedding(x)
        list_encoding = []
        for layer in self.layers_encoding:
            encoding = layer(encoding)
            list_encoding.append(encoding)
        output = self._linear2(encoding)
        output = self._linear(output)
        #output = torch.sigmoid(output)
        return output, list_encoding