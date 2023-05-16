import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
import data.dcrnn_utils as dcrnn_utils
from models.encoder import *
#from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.gnn import gcn, gcn_gwnet, gcn_gcnm_dynamic, spatialGCN
from models.memoryModule import LocalFeatureModule, MemoryModule
from models.modules import GraphAttentionLayer, SpatioTemporalAttention, TemporalConvolution

class TemporalConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=(kernel_size, 1), stride=(stride, 1),
                              padding=((kernel_size - 1) // 2, 0))

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x

class GWNet(nn.Module):
    def __init__(self, num_nodes, num_timesteps_input, num_timesteps_output):
        super(GWNet, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        
        self.spatial_conv = nn.Conv2d(in_channels=1, out_channels=64,
                                      kernel_size=(1, num_nodes))
        self.temporal_conv1 = TemporalConvolutionLayer(in_channels=64, out_channels=64,
                                                       kernel_size=3, stride=1)
        self.temporal_conv2 = TemporalConvolutionLayer(in_channels=64, out_channels=64,
                                                       kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_timesteps_output)

    def forward(self, x):
        # Input shape: (batch_size, num_nodes, num_timesteps_input)
        # Output shape: (batch_size, num_timesteps_output)
        x = x.permute(0, 2, 1).unsqueeze(1)
        # Shape: (batch_size, 1, num_nodes, num_timesteps_input)
        x = self.spatial_conv(x)
        # Shape: (batch_size, 64, num_nodes, num_timesteps_input)
        x = x.permute(0, 2, 3, 1)
        # Shape: (batch_size, num_nodes, num_timesteps_input, 64)
        x = self.temporal_conv1(x)
        # Shape: (batch_size, num_nodes, num_timesteps_input, 64)
        x = self.temporal_conv2(x)
        # Shape: (batch_size, num_nodes, num_timesteps_input, 64)
        x = x.mean(dim=1)
        # Shape: (batch_size, num_timesteps_input, 64)
        x = x.reshape(-1, 64)
        # Shape: (batch_size*num_timesteps_input, 64)
        x = F.relu(self.fc1(x))
        # Shape: (batch_size*num_timesteps_input, 64)
        x = self.fc2(x)
        # Shape: (batch_size*num_timesteps_input, num_timesteps_output)
        x = x.reshape(-1, self.num_timesteps_input, self.num_timesteps_output)
        # Shape: (batch_size, num_timesteps_input, num_timesteps_output)
        x = x.mean(dim=1)
        # Shape: (batch_size, num_timesteps_output)
        return x
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class DMSTGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 out_dim=12, residual_channels=16, dilation_channels=16, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, days=288, dims=40, order=2, in_dim=9, normalization="batch"):
        super(DMSTGCN, self).__init__()
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p1 = nn.Parameter(torch.randn(days, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p2 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2p3 = nn.Parameter(torch.randn(num_nodes, dims).to(device), requires_grad=True).to(device)
        self.nodevec_a2pk = nn.Parameter(torch.randn(dims, dims, dims).to(device), requires_grad=True).to(device)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv1d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a2p.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))

        self.relu = nn.ReLU(inplace=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, inputs, ind):
        """
        input: (B, F, N, T)
        """
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [1]])
        skip = 0

        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        adp_a2p = self.dgconstruct(self.nodevec_a2p1[ind], self.nodevec_a2p2, self.nodevec_a2p3, self.nodevec_a2pk)

        new_supports = [adp]
        new_supports_a = [adp_a]
        new_supports_a2p = [adp_a2p]

        for i in range(self.blocks * self.layers):
            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # tcn for auxiliary part
            residual_a = x_a
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            # dynamic graph convolutions
            x = self.gconv[i](x, new_supports)
            x_a = self.gconv_a[i](x_a, new_supports_a)

            # multi-faceted fusion module
            x_a2p = self.gconv_a2p[i](x_a, new_supports_a2p)
            x = x_a2p + x

            # residual and normalization
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x = x + residual[:, :, :, -x.size(3):]
            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)

        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class STF_InformerStack(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 spatio_temporal_kernel_size, dropout, adj_mtx, static_node_emb):
        super(STF_InformerStack, self).__init__()
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.spatio_temporal_kernel_size = spatio_temporal_kernel_size
        
        # Encoder
        self.encoder_layer = nn.ModuleList([GraphAttentionLayer(d_model, adj_mtx, dropout) for _ in range(num_encoder_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.temporal_conv = TemporalConvolution(d_model, dim_feedforward, dropout)
        self.encoder_attn = SpatioTemporalAttention(d_model, spatio_temporal_kernel_size, dropout)
        self.encoder_dropout = nn.Dropout(dropout)
        
        # Decoder
        self.decoder_layer = nn.ModuleList([GraphAttentionLayer(d_model, adj_mtx, dropout) for _ in range(num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_attn = SpatioTemporalAttention(d_model, spatio_temporal_kernel_size, dropout)
        self.static_node_emb = nn.Parameter(torch.Tensor(static_node_emb))
        
        # Output
        self.output_layer = nn.Linear(d_model, 1)
        
        # Initialize static node embedding
        nn.init.xavier_uniform_(self.static_node_emb)
    
    def forward(self, x):
        # Encoder
        x_encoder = x[:, :-1, :]
        x_static = x[:, -1, :]
        for layer in self.encoder_layer:
            x_encoder = layer(x_encoder)
        x_encoder = self.encoder_norm(x_encoder)
        x_encoder = self.temporal_conv(x_encoder)
        x_encoder = self.encoder_attn(x_encoder)
        x_encoder = self.encoder_dropout(x_encoder)
        
        # Decoder
        x_decoder = torch.cat([x_static.unsqueeze(1), x_encoder], dim=1)
        for layer in self.decoder_layer:
            x_decoder = layer(x_decoder)
        x_decoder = self.decoder_norm(x_decoder)
        x_decoder = self.decoder_attn(x_decoder)
        
        # Output
        x_out = self.output_layer(x_decoder)
        
        return x_out.squeeze()

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
class GCNM(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        """

        full_data: full dataset including dateTime
        in_dim: the input data dimension (i.e., node numbers)
        """
        super(GCNM, self).__init__()
        self.local_feature_model = LocalFeatureModule(num_nodes)
        self.memory_model = MemoryModule(in_dim, residual_channels)

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()


        ##s to check if we still need "start_conv"???
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn_gwnet(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


    def forward(self, input, x_hist):
        """

        :param input: (B, 8, L, D)
        :param x_hist: (B, n*tau, L, D)
        :return: e: enrichied traffic embedding (B, L, D)
        """

        z = self.local_feature_model(input) #(B, L, D)
        z = torch.unsqueeze(z, dim=-1) # (B, L, D) -> (B, L, D, 1)
        x_hist = torch.unsqueeze(x_hist, dim=-1)#(B, n*tau, L, D, 1)
        x_hist = x_hist.transpose(1, 2).contiguous() #(B, L, n*tau, D, F)

        #(B, L, D, F), (B, L, n*tau, D, F)

        e = self.memory_model(z, x_hist) # (B, L, D, F), (B, L, n*tau, D, F) -> (B, F', L, D)

        input = e.permute(0, 1, 3, 2).contiguous()  #(B, F', D, L)

        """
                # the input is from the enriched temporal embedding
                # input: temporal embedding (N, 1, D, L)
                """

        in_len = input.size(3)  # (N, F, D, L), here F=1
        if in_len < self.receptive_field:  # receptive_filed = 12 + 1
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))  # (N, F, D, L+1)
        else:
            x = input
        #x = self.start_conv(x)  # kernel=(1,1), (N, 1, D, L+1) -> (N, 1, D, L+1)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)  # kernel=(1, 2)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # kernel=(1,2)
            gate = torch.sigmoid(gate)
            # x=filter=gate: (B, residual_size, D, F)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # x: (B, residual_size, D, F)
                    #print("input.shape 1 is {}".format(x.size()))
                    x = self.gconv[i](x, new_supports)
                    #print("input.shape 2 is {}".format(x.size()))
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [N, L, D, 1]
        x = torch.squeeze(x, dim=-1)  # [N, L, D]

        return x.contiguous()

class GCNMdynamic(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        """

        full_data: full dataset including dateTime
        in_dim: the input data dimension (i.e., node numbers)
        """
        super(GCNMdynamic, self).__init__()
        self.local_feature_model = LocalFeatureModule(num_nodes)
        self.memory_model = MemoryModule(in_dim, residual_channels)

        self.num_nodes = num_nodes
        self.device = device
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        ##s to check if we still need "start_conv"???
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 2

        #parameters for initializing the static node embeddings
        node_dim = residual_channels
        self.alpha = 3
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        self.GCN1_1 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN1_2 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN2_1 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        self.GCN2_2 = gcn_gwnet(c_in=residual_channels,c_out=residual_channels,
                                dropout=self.dropout,support_len=1)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn_gcnm_dynamic(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        if out_dim > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, out_dim), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, out_dim-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

    def preprocessing(self, adj):
        #adj: (B, L, D, D)
        adj = adj + torch.eye(self.num_nodes).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return adj

    def forward(self, input, x_hist):
        """

        :param input: (B, 8, L, D)
        :param x_hist: (B, n*tau, L, D)
        :return: e: enrichied traffic embedding (B, L, D)
        """

        z = self.local_feature_model(input) #(B, L, D)
        z = torch.unsqueeze(z, dim=-1) # (B, L, D) -> (B, L, D, 1)
        x_hist = torch.unsqueeze(x_hist, dim=-1)#(B, n*tau, L, D, 1)
        x_hist = x_hist.transpose(1, 2).contiguous() #(B, L, n*tau, D, F)

        #(B, L, D, F), (B, L, n*tau, D, F)

        e = self.memory_model(z, x_hist) # (B, L, D, F), (B, L, n*tau, D, F) -> (B, residual_channels, L, D)

        input = e.permute(0, 1, 3, 2).contiguous()  #(B, F', D, L)

        """
                # the input is from the enriched temporal embedding
                # input: temporal embedding (N, 1, D, L)
                """

        in_len = input.size(3)  # (N, F, D, L), here F=1
        if in_len < self.receptive_field:  # receptive_filed = 12 + 1
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))  # (N, residual_channels, D, L+1)
        else:
            x = input
        #x = self.start_conv(x)  # kernel=(1,1), (N, 1, D, L+1) -> (N, 1, D, L+1)
        #skip = 0
        skip = self.skip0(x)
        # calculate the current adaptive adj matrix once per iteration
        '''new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        '''

        # x: (N, residual_channels, D, L), support[i]: (D, D)
        nodevecInit_1 = self.emb1(self.idx)  # (D, node_dim=residual_channels)
        nodevecInit_2 = self.emb2(self.idx)  # (D, node_dim=residual_channels)
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)  # kernel=(1, 2)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # kernel=(1,2)
            gate = torch.sigmoid(gate)
            # x=filter=gate: (B, residual_size, D, F)
            x = filter * gate

            # ***************** construct dynamic graphs from e ***************** #
            # print("x.size: {}, support0: {}, support1: {}".format(x.size(), self.supports[0].size(), self.supports[1].size()))
            '''filter1 = self.GCN1_1(x, [self.supports[0]]) + self.GCN1_2(x, [
                self.supports[1]])  # (N, residual_channels, D, L)
            filter2 = self.GCN2_1(x, [self.supports[0]]) + self.GCN2_2(x, [
                self.supports[1]])  # (N, residual_channels, D, L)'''
            filter1 = self.GCN1_1(x, [self.supports[0]]) # (N, residual_channels, D, L)
            filter2 = self.GCN2_1(x, [self.supports[1]]) # (N, residual_channels, D, L)
            filter1 = filter1.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
            filter2 = filter2.permute((0, 3, 2, 1)).contiguous()  # (N, L, D, residual_channels)
            nodevec1 = torch.tanh(self.alpha * torch.mul(nodevecInit_1, filter1))  # (N, L, D, residual_channels)
            nodevec2 = torch.tanh(self.alpha * torch.mul(nodevecInit_2, filter2))

            # objective: construct "support/A" with size (B, D, D, L)
            a = torch.matmul(nodevec1, nodevec2.transpose(2, 3)) - torch.matmul(
                nodevec2, nodevec1.transpose(2, 3))  # (B, L, D, D)
            adj = F.relu(torch.tanh(self.alpha * a))
            mask = torch.zeros(adj.size(0), adj.size(1), adj.size(2), adj.size(3)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(20, -1)
            mask.scatter_(-1, t1, s1.fill_(1))
            adj = adj * mask

            adp = self.preprocessing(adj)
            adpT = self.preprocessing(adj.transpose(2, 3))
            adp = adp.permute((0, 2, 3, 1)).contiguous()  # (B, D, D, L)
            adpT = adpT.permute((0, 2, 3, 1)).contiguous()
            #new_supports = [adp, adpT, self.supports[0], self.supports[1]]  # dynamic and pre-defined graph
            new_supports = [adp, adpT]

            # parametrized skip connection
            #x = F.dropout(x, self.dropout)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # x: (B, residual_size, D, F)
                    #print("input.shape 1 is {}".format(x.size()))
                    x = self.gconv[i](x, new_supports)
                    #print("input.shape 2 is {}".format(x.size()))
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        skip = self.skipE(x) + skip

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x)) # [N, skip_channels, D, 1] -> [N, end_channels, D, 1]
        x = self.end_conv_2(x)  # [N, end_channels, D, 1] -> [N, L, D, 1]
        x = torch.squeeze(x, dim=-1)  # [N, L, D]

        return x.contiguous()