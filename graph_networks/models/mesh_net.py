import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add

from graph_networks.parameters import Parameters
from .models_base import Parallel


def Create_Parallel(num_channels, in_size, out_size):
    modules= [nn.Sequential(nn.Linear(in_size, out_size), 
                            nn.LeakyReLU(0.2),
                            nn.Linear(out_size, out_size), 
                            nn.LeakyReLU(0.2),
                            nn.Linear(out_size, out_size, bias=False), 
                            nn.LayerNorm(out_size)) for _ in range(num_channels)]
    return Parallel(*modules)



class MeshCNN(nn.Module):
    def __init__(self, in_channel, in_nodes, out_nodes=1, num_nodes=4981) -> None:
        super().__init__()
        self.in_channel= in_channel
        self.num_nodes = num_nodes
        self.model= Create_Parallel(in_channel, in_nodes, out_nodes)

    def forward(self, x, elem_conn):
        x= x.reshape(-1, self.num_nodes, self.in_channel) #todo check with pyg on this
        x= x[:, elem_conn]
        x= self.model(x).squeeze(2)
        return x.reshape(-1, self.in_channel)

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv= MeshCNN(in_channel, 4, 1)
        self.expander= nn.Sequential(nn.Linear(in_channel, out_channel),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(out_channel, out_channel), 
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(out_channel, out_channel, bias=False), 
                                     nn.LayerNorm(out_channel))
                                     
    def forward(self, x, elem_conn):
        x= self.conv(x, elem_conn)
        return self.expander(x)

class EdgeModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.edge_model = nn.Sequential(nn.Linear(2*in_channel, out_channel),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(out_channel, out_channel), 
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(out_channel, out_channel, bias=False), 
                                        nn.LayerNorm(out_channel))
        
    def forward(self, x, edge_index):
        # edge_index: [2, E], where E is the number of edges.
        # x: [N, F_x]
        # message: [E, F_e]
        
        src, dest= edge_index[0], edge_index[1]
        message = torch.cat((x[dest], x[src]), dim=-1)
        message = self.edge_model(message)
        return message


class NodeModel(nn.Module):
    def __init__(self, in_channel, out_channel, msg_channel, reduction='add'):
        super().__init__()
        self.reduciton= reduction
        self.node_model = nn.Sequential(nn.Linear(in_channel + msg_channel, out_channel),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(out_channel, out_channel), 
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(out_channel, out_channel, bias=False), 
                                        nn.LayerNorm(out_channel))

    def forward(self, x, edge_index, message):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # messsage: [E, F_e]

        node_dim= 0 if x.dim()==2 else 1 
        N= x.size(node_dim)

        src, dest= edge_index[0], edge_index[1]

        msg_aggr = scatter(message, dest, dim=node_dim, dim_size=N, reduce=self.reduciton)
        out = torch.cat((x, msg_aggr), dim=-1)
        out= self.node_model(out)
        return out


class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model, node_model):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index):
        
        message = self.edge_model(x, edge_index)
        out = self.node_model(x, edge_index, message)
        return out


class Processor(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, msg_channel, depth):
        super().__init__()

        self.graphnet_blocks = nn.ModuleList()
        self.graphnet_blocks.append(self.meta_layer(in_channel, hidden_channel, 
                                                    msg_channel))
        for i in range(1, depth-1):
            self.graphnet_blocks.append(self.meta_layer(hidden_channel, hidden_channel, 
                                                        msg_channel))

        self.graphnet_blocks.append(self.meta_layer(hidden_channel, out_channel, 
                                                    msg_channel))

    def forward(self, x, edge_index):
        for graphnet_block in self.graphnet_blocks:
            x = graphnet_block(x, edge_index)
        return x

    @staticmethod
    def meta_layer(in_channel, out_channel, msg_channel):
        return MetaLayer(EdgeModel(in_channel, msg_channel),
                         NodeModel(in_channel, out_channel, 
                                   msg_channel))



class MeshUpSample(nn.Module):
    def __init__(self, in_channel, num_nodes=4981) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.num_nodes = num_nodes
        self.model = Create_Parallel(in_channel, 1, 4)

    def forward(self, x, elem_conn):
        n_e= elem_conn.size(0)
        x= x.reshape(-1, n_e, self.in_channel)
        x = self.model(x.unsqueeze(2))

        bs = x.size(0)
        out = torch.zeros(bs, self.num_nodes, self.in_channel, dtype=x.dtype, device=x.device)
        for i in range(4):
            out.index_add_(1, elem_conn.T[i], x[:, :, i, :])
        return out.reshape(-1, self.in_channel)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.deconv= MeshUpSample(in_channel)
        self.contracter= nn.Sequential(nn.Linear(in_channel, out_channel),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(out_channel, out_channel), 
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(out_channel, out_channel, bias=False))
                                     
    def forward(self, x, elem_conn):
        x= self.deconv(x, elem_conn)
        return self.contracter(x)
        
        
class MeshNet(nn.Module):
    def __init__(self, params: Parameters) -> None:
        super().__init__() 
        h_params= params.hyper

        self.elem_conn= torch.from_numpy(np.load(params.data.data_dir+ r"/Daten/node_element_id_81000001.npy")).to(h_params.device)
        self.elem_conn-= self.elem_conn.min()

        self.encoder= Encoder(h_params.in_channel, h_params.hidden_channel)
        self.processor= Processor(h_params.hidden_channel, h_params.hidden_channel,
                                  h_params.hidden_channel, h_params.msg_channel, h_params.depth)
        self.decoder= Decoder(h_params.hidden_channel, h_params.out_channel)
    
    def forward(self, data):
        x= data.x
        x= self.encoder(x, self.elem_conn)
        x= self.processor(x, data.elem_index)
        x= self.decoder(x, self.elem_conn)
        return x 
