import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_stack = self._build_encoder_stack()
        
    def _build_encoder_stack(self):
        return nn.ModuleList([
            self.EncoderLayer(self.hidden_dim, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])
    
    def forward(self, x, mask):
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x, mask)
        return x
    
    class EncoderLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads, dropout):
            super(Encoder.EncoderLayer, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.conv_layers = self._build_conv_layers()
            self.norm_layer = nn.LayerNorm(hidden_dim)
            self.dropout_layer = nn.Dropout(dropout)
            
        def _build_conv_layers(self):
            return nn.ModuleList([
                self.ConvLayer(self.hidden_dim, self.num_heads)
                for _ in range(self.num_heads)
            ])
        
        def forward(self, x, mask):
            attn_outputs = []
            for conv_layer in self.conv_layers:
                attn_output = conv_layer(x, mask)
                attn_outputs.append(attn_output)
            attn_outputs = torch.cat(attn_outputs, dim=-1)
            x = self.norm_layer(x + self.dropout_layer(attn_outputs))
            return x
        
        class ConvLayer(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                super(Encoder.EncoderLayer.ConvLayer, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_heads = num_heads
                self.q_layer = nn.Linear(hidden_dim, hidden_dim)
                self.k_layer = nn.Linear(hidden_dim, hidden_dim)
                self.v_layer = nn.Linear(hidden_dim, hidden_dim)
                
            def forward(self, x, mask):
                q = self.q_layer(x)
                k = self.k_layer(x)
                v = self.v_layer(x)
                attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_output = torch.matmul(attn_probs, v)
                return attn_output
