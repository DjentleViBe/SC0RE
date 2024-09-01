"""Encoder part of the transformer
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

def scaled_dot_product(q_val, k_val, v_val, mask):
    """Required for attention
    q_val : what am I looking for
    k_val : what do I have to offer
    v_val : What I actually offer
    """
    # q,k,v = 30 x 8 x 10 x 64
    d_k = q_val.size()[-1] # 64
    # x : [batch, sequence_length, embed]
    # x.transpose(-1,-2) : [batch, embed, sequence_length]
    # sqrt(d_k) is required to ensure the variance ahs a mean standard
    # deviation of 1 for stable back propogation
    # 30 x 8 x 10 x 64 matmul 30 x 8 x 64 x 10 = 30 x 8 x 10 x 10 ( pre-cursor
    # to self attention matrix)
    scaled = (torch.matmul(q_val, k_val.transpose(-1, -2)) / math.sqrt(d_k))
    if mask is not None:
        scaled += mask
    # to get probabilities apply softmax
    attention = F.softmax(scaled, dim = -1)
    # 30 x 8 x 200 x 64
    values = torch.matmul(attention, v_val)
    return values, attention

class MultiHeadAttentionAPE(nn.Module):
    """Multi head self attention
    """
    def __init__(self, device, d_model, seq_length, num_heads) -> None:
        super().__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.seq_length = seq_length
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)   # 512 x 1536
        self.qkv_layer.to(device)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.linear_layer.to(device)

    def forward(self, x_val, mask=None):
        """Forward layer
        """
        # 30 x 10 x 32
        batch_size, sequence_length, _ = x_val.size()
        # 30 x 10 x 96
        qkv = self.qkv_layer(x_val)
        # 30 x 10 x 8 x 12, breaking q,k v into 8 heads
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        # 30 x 8 x 10 x 12
        qkv = qkv.permute(0, 2, 1, 3)
        # break tensor into respective parts, each are 30 x 8 x 10 x 12
        # batch x head x seq x (3 * (embed / head))
        q_val, k_val, v_val = qkv.chunk(3, dim = -1)
        # 30 x 8 x 10 x 32, 30 x 8 x 10 x 10
        values, _ = scaled_dot_product(q_val, k_val, v_val, mask)
        # 30 x 10 x 32
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # 30 x 10 x 1
        values = self.linear_layer(values)
        return values

class LayerNormalization(nn.Module):
    """Layer normalisation
    """
    def __init__(self, device, parameters_shape, eps=1e-5) -> None:
        super().__init__()
        # parameters_shape gives the dimension along which layer normalisation is to be performed
        self.parameters_shape = parameters_shape # 512
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)).to(device) # learnable parameter
        self.beta = nn.Parameter(torch.zeros(parameters_shape)).to(device) # learnable parameter

    def forward(self, inputs): # 30 x 10 x 512
        """Forward prop
        """
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # -1
        mean = inputs.mean(dim=dims, keepdim=True) # 30 x 10 x 1
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # 30 x 10 x 1
        std = (var + self.eps).sqrt() # 30 x 10 x 1
        y_val = (inputs - mean) / std # 30 x 10 x 512
        out = self.gamma * y_val + self.beta # 30 x 10 x 512
        return out

class PositionwiseFeedForward(nn.Module):
    """Feed forward layer
    """
    def __init__(self, device, d_model, hidden, dropprob=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 x 2048
        self.linear1.to(device)
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.linear2.to(device)
        self.relu = nn.ReLU()
        self.relu.to(device)
        self.dropout = nn.Dropout(dropprob)
        self.dropout.to(device)

    def forward(self, x_val):   # 30 x 10 x 512
        """Forward pass
        """
        x_val = self.linear1(x_val) # 30 x 10 x 2048
        x_val = self.relu(x_val)    # 30 x 10 x 2048
        x_val = self.dropout(x_val) # 30 x 10 x 2048
        x_val = self.linear2(x_val) # 30 x 10 x 512

        return x_val

class EncoderLayerAPE(nn.Module):
    """Main encoder layer
    """
    def __init__(self, device, d_model, ffn_hidden, seq_length, num_heads, drop_prob) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionAPE(device, d_model, seq_length, num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout1.to(device)
        self.norm1 = LayerNormalization(device, parameters_shape=[d_model])
        self.norm1.to(device)
        self.ffn = PositionwiseFeedForward(device, d_model, ffn_hidden, drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout2.to(device)
        self.norm2 = LayerNormalization(device, parameters_shape=[d_model])

    def forward(self, x_val):
        """Forward layer
        """
        residual_x = x_val # 30 x 10 x 512
        x_val = self.attention(x_val, mask = None) # 30 x 200 x 512
        x_val = self.dropout1(x_val) # 30 x 10 x 512
        x_val = self.norm1(x_val + residual_x) # 30 x 10 x 512
        residual_x = x_val # 30 x 10 x 512
        x_val = self.ffn(x_val) # 30 x 10 x 512
        x_val = self.dropout2(x_val) # 30 x 10 x 512
        x_val = self.norm2(x_val + residual_x)  # 30 x 10 x 512
        return x_val

class EncoderAPE(nn.Module):
    """Main Encoder class
    """
    def __init__(self, device, d_model, ffn_hidden, seq_length, \
                 num_heads, drop_prob, num_layers) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayerAPE(device, d_model, ffn_hidden, \
                                seq_length, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x_val):
        """Forward layer
        """
        x_val = self.layers(x_val)
        return x_val