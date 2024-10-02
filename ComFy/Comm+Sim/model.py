# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SGConv
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout, GELU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


# class ResidualModuleWrapper(nn.Module):
#     def __init__(self, module, normalization, dim, **kwargs):
#         super().__init__()
#         self.normalization = normalization(dim)
#         self.module = module(dim=dim, **kwargs)

#     def forward(self, graph, x):
#         x_res = self.normalization(x)
#         x_res = self.module(graph, x_res)
#         x = x + x_res

#         return x


# class FeedForwardModule(nn.Module):
#     def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
#         super().__init__()
#         input_dim = int(dim * input_dim_multiplier)
#         hidden_dim = int(dim * hidden_dim_multiplier)
#         self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
#         self.dropout_1 = nn.Dropout(p=dropout)
#         self.act = nn.GELU()
#         self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
#         self.dropout_2 = nn.Dropout(p=dropout)

#     def forward(self, graph, x):
#         x = self.linear_1(x)
#         x = self.dropout_1(x)
#         x = self.act(x)
#         x = self.linear_2(x)
#         x = self.dropout_2(x)

 #       return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5,
                 dropout=0.5, save_mem=False, use_bn=True):
        super().__init__()
        torch.manual_seed(12345)

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATv2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=16):
        torch.manual_seed(12345)
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=heads)

    def forward(self, x, edge_index,p=0.0):
        h = F.dropout(x, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features,num_classes,hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)


    def forward(self, x, edge_index,p=0.0):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)
        return x


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

class SGC(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = SGConv(
            in_channels=num_features,
            out_channels=num_classes,
            K=1,
            cached=True,
        )

    def forward(self,x,edge_index,p=0.0):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


# class CustomGCN(nn.Module):
#     def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
#         super().__init__()
#         self.feed_forward_module = FeedForwardModule(dim=dim,
#                                                      hidden_dim_multiplier=hidden_dim_multiplier,
#                                                      dropout=dropout)

#     def forward(self, graph, x):
#         degrees = graph.out_degrees().float()
#         degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
#         norm_coefs = 1 / degree_edge_products ** 0.5

#         x = ops.u_mul_e_sum(graph, x, norm_coefs)

#         x = self.feed_forward_module(graph, x)

#         return x
