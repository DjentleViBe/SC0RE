"""decoder part of the transformer
"""
# import torch
from torch import nn
from _encoder.encoder import LayerNormalization, PositionwiseFeedForward,\
                    MultiHeadAttentionAPE

class DecoderLayerAPE(nn.Module):
    """Decoder layer
    """
    def __init__(self, device, d_model, d_vocab, seq_length, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttentionAPE(device, d_model, seq_length, num_heads)
        self.layer_norm1 = LayerNormalization(device, parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(device, d_model, ffn_hidden, drop_prob)
        self.layer_norm3 = LayerNormalization(device, parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.linear_layer = nn.Linear(d_model, d_model)
        self.vocab_layer = nn.Linear(d_model, d_vocab)

    def forward(self, y_val, decoder_mask):
        """Forward prop
        """
        _y = y_val.clone()
        y_val = self.self_attention(y_val, mask=decoder_mask)
        y_val = self.dropout1(y_val)
        y_val = self.layer_norm1(y_val + _y)

        # _x = x_val.clone()
        _y = y_val.clone()
        y_val = self.ffn(y_val)
        y_val = self.dropout3(y_val)
        y_val = self.layer_norm3(y_val + _y)
        y_val = self.linear_layer(y_val)
        y_val = self.vocab_layer(y_val)
        return y_val

class SequentialDecoder(nn.Sequential):
    """To pass more than one parameter in sequential
    """
    def forward(self, *inputs):
        """forward pass for custom seq decoder
        """
        yseq, mask = inputs
        for module in self._modules.values():
            yseq = module(yseq, mask)
        return yseq

class DecoderAPE(nn.Module):
    """Main decoder class
    """
    def __init__(self, device, d_model, d_vocab, ffn_hidden, seq_length, num_heads, drop_prob, num_layers) -> None:
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayerAPE(device, d_model, d_vocab, seq_length,ffn_hidden,num_heads,drop_prob)\
                                      for _ in range(num_layers)])

    def forward(self, y_val, decoder_mask):
        """Forward layer
        """
        x_val = self.layers(y_val, decoder_mask)
        return x_val