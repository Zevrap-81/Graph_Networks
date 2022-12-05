from collections import OrderedDict
import functools

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter, scatter_add
# import trainer as torch_trainer

from graph_networks.parameters import Hyperparameters

class LazyMLP(nn.Module):
    
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            if index < (num_layers - 1):
                self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
                self._layers_ordered_dict["relu_" + str(index)] = nn.LeakyReLU(0.2)
            else:
                self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size, bias=False)

        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        y = self.layers(input)
        return y


class EdgeModel(nn.Module):
    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self.edge_models= nn.ModuleDict({
                                            'mesh_edge': make_mlp(latent_size),
                                            'world_edge': make_mlp(latent_size)
                                        })
        
    def forward(self, data:Data):
        # s, r: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]

        s, r= data.mesh_edge_index
        x= torch.cat((data.x[s], data.x[r], data.mesh_edge_attr), dim=-1)
        data.mesh_edge_attr+= self.edge_models['mesh_edge'](x)

        s,r= data.world_edge_index
        x= torch.cat((data.x[s], data.x[r], data.world_edge_attr), dim=-1)
        data.world_edge_attr+= self.edge_models['world_edge'](x)

        return data


class NodeModel(nn.Module):
    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.node_mlp = make_mlp(output_size)

    def forward(self, data:Data):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]

        N = data.x.size(0)

        x= []
        x.append(data.x) 
        x.append(scatter(data.mesh_edge_attr, data.mesh_edge_index[1], dim=0, reduce='mean', dim_size=N))
        x.append(scatter(data.world_edge_attr, data.world_edge_index[1], dim=0, reduce='mean', dim_size=N))
        x= torch.cat(x, dim=-1)

        x= self.node_mlp(x)
        data.x+= x
        return data


class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model


    def forward(self, data:Data):
        if self.edge_model is not None:
            data = self.edge_model(data)
        
        if self.node_model is not None:
            data = self.node_model(data)

        if self.global_model is not None:
            data = self.global_model(data)

        return data


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._latent_size = latent_size
        self.node_model = make_mlp(latent_size)
        self.edge_models= nn.ModuleDict({
                                            'mesh_edge': make_mlp(latent_size),
                                            'world_edge': make_mlp(latent_size)
                                        })

    def forward(self, data:Data):
        
        data.x = self.node_model(data.x)
        
        data.mesh_edge_attr= self.edge_models['mesh_edge'](data.mesh_edge_attr)
        data.world_edge_attr= self.edge_models['world_edge'](data.world_edge_attr)

        return data


class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection
    (features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent 
    graph will be feed into original processor
    message_passing_steps - epochs
    '''

    def __init__(self, make_mlp, latent_size, message_passing_steps):
        super().__init__()
        self.graphnet_blocks = nn.ModuleList()

        for _ in range(message_passing_steps):
            self.graphnet_blocks.append(MetaLayer(EdgeModel(make_mlp, latent_size),
                                                  NodeModel(make_mlp, latent_size))
                                        )

    def forward(self, data:Data):
        for graphnet_block in self.graphnet_blocks:
            data = graphnet_block(data)
        return data


class Decoder(nn.Module):
    """Decodes node features from graph."""
    # decoder = self._make_mlp(self._output_size, layer_norm=False)
    # return decoder(graph['node_features'])

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.node_model = make_mlp(output_size)

    def forward(self, data:Data):
        # todo remove these hardcoded dependencies
        x= self.node_model(data.x).reshape(-1, 19674, 4)
        x= x[:, 0:4981, :].reshape(-1, 4)

        # x= self.node_model(data.x).reshape(1, -1, 4)
        # x= x[:, 0:4981, :].reshape(-1, 4)
        return x


class MeshGraphNet_legacy(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self, params: Hyperparameters):
        super().__init__()

        self.hidden_size = params.hidden_channel
        self.output_size= params.out_channel
        self._num_layers= params.depth
        message_passing_steps= params.message_passing_steps

        self.encoder = Encoder(make_mlp=self._make_mlp,
                               latent_size=self.hidden_size)

        self.processor = Processor(make_mlp=self._make_mlp,
                                   latent_size=self.hidden_size,
                                   message_passing_steps=message_passing_steps)

        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self.output_size)


    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self.hidden_size] * (self._num_layers - 1) + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, data:Data):
        """Encodes and processes a multigraph, and returns node features."""

        data = self.encoder(data)
        data = self.processor(data)
        return self.decoder(data)
