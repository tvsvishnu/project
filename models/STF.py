import torch
import torch.nn as nn
from models.modules import GraphAttentionLayer, SpatioTemporalAttention, TemporalConvolution

class STF_Informer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Define the encoder graph attention layers
        self.encoder_gat_layers = nn.ModuleList()
        for i in range(num_encoder_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim * num_heads
            self.encoder_gat_layers.append(GraphAttentionLayer(in_dim, out_dim, num_heads))

        # Define the spatio-temporal attention layer
        self.st_attention = SpatioTemporalAttention(hidden_dim, num_heads)

        # Define the decoder graph attention layers
        self.decoder_gat_layers = nn.ModuleList()
        for i in range(num_decoder_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim * num_heads
            self.decoder_gat_layers.append(GraphAttentionLayer(in_dim, out_dim, num_heads))

        # Define the temporal convolution layer
        self.temporal_conv = TemporalConvolution(hidden_dim * num_heads, output_dim)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Encoder stage
        encoder_outputs = []
        for i, gat_layer in enumerate(self.encoder_gat_layers):
            if i == 0:
                encoder_input = x
            else:
                encoder_input = torch.cat(encoder_outputs, dim=-1)

            encoder_output = gat_layer(encoder_input, adj)
            encoder_output = self.dropout(encoder_output)
            encoder_outputs.append(encoder_output)

        # Spatio-temporal attention
        st_input = encoder_outputs[-1]
        st_output = self.st_attention(st_input)
        st_output = self.dropout(st_output)

        # Decoder stage
        decoder_outputs = []
        for i, gat_layer in enumerate(self.decoder_gat_layers):
            if i == 0:
                decoder_input = st_output
            else:
                decoder_input = torch.cat(decoder_outputs, dim=-1)

            decoder_output = gat_layer(decoder_input, adj)
            decoder_output = self.dropout(decoder_output)
            decoder_outputs.append(decoder_output)

        # Temporal convolution
        output = self.temporal_conv(decoder_outputs[-1])
        output = self.dropout(output)

        # Return the output
        return output
