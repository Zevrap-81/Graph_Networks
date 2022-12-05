import enum 
import os.path as osp

import numpy as np 
import pandas as pd 
import torch 
import torch.nn.functional as F 
from torch_geometric.data import Dataset, Data

from tqdm import trange

from graph_networks.parameters import DataParameters
from graph_networks.common import quads_to_edges, kdtree_nearest_graph
from graph_networks.normalizer import Normalizer

class NodeType(enum.IntEnum):
  PLATE = 0
  PUNCH = 1
  HOLDER = 2
  DIE = 3
  SIZE = 4


class Mesh_Dataset(Dataset):
    def __init__(self, data_params: DataParameters=None):
        self.data_params= data_params
        super().__init__(data_params.data_dir) #pre_transform

    def process(self):
        connectivity = np.load(osp.join(self.raw_dir, r"node_element_id_81000001.npy"))
        connectivity-= connectivity.min()
        mesh_edges= torch.from_numpy(quads_to_edges(connectivity))
        world_edges, pos, node_types= self.get_world_space()

        one_hot_node_type = F.one_hot(node_types, NodeType.SIZE)
        x= torch.hstack((one_hot_node_type, self.normalize_(pos)))

        doe_= pd.read_csv(osp.join(self.root, r"Experiments_1_no_error.csv"), sep=';', decimal=',')
        doe_= doe_.iloc[:, 1:6].to_numpy().astype(np.float32) #F_press;fs;h_BT;s_0;sfo;sigy
        doe_= self.normalize_(torch.from_numpy(doe_), name='doe')

        rel_mesh= self.rel_pos(mesh_edges, pos)
        rel_world= [self.rel_pos(edge_index, pos) for edge_index in world_edges] 
        
        pos_plate = torch.from_numpy(np.load(self.raw_dir+"/pid_81000001_run_1_stage_1.npy")[:, 0:3, 0])
        world_edges= torch.hstack(world_edges)
        y_norm= Normalizer.load_state(osp.join(self.raw_dir, r"processed/output_norm.pkl"))
        indices_without_outliers= np.delete(np.arange(0, self.data_params.num_samples, dtype=np.int64), self.data_params.outliers_list)
        total_size= self.data_params.total_size
        
        for i in trange(total_size, unit="samples", colour='red'):
            d= doe_[indices_without_outliers[i]]
            e_plate= torch.ones(len(rel_mesh), 2)* torch.unsqueeze(d[[3,4]], 0) #s_0;sfo
            e_plate= torch.hstack((rel_mesh, e_plate))

            e_punch= torch.ones(len(rel_world[0]), 2)* torch.unsqueeze(d[[2,1]], 0) #h_BT;fs
            e_punch= torch.hstack((rel_world[0], e_punch))
            
            e_holder= torch.ones(len(rel_world[1]), 2)* torch.unsqueeze(d[[0,1]], 0) #F_press;fs
            e_holder= torch.hstack((rel_world[1], e_holder))
            
            e_die= torch.ones(len(rel_world[2]), 2)* torch.unsqueeze(d[[2,1]], 0) #h_BT;fs
            e_die= torch.hstack((rel_world[2], e_die))

            edge_attr= torch.vstack((e_punch, e_holder, e_die))

            y= torch.from_numpy(np.load(osp.join(self.raw_dir, rf"processed/y_{i}.npy"))) #, allow_pickle=True
            y= y_norm.transform_(y)

            data= Data(x=x, y=y, mesh_edge_index=mesh_edges, mesh_edge_attr=e_plate, pos=pos_plate,
                       world_edge_index=world_edges, world_edge_attr=edge_attr, num_nodes=len(pos))


            torch.save(data, osp.join(self.processed_dir, rf"data_{i}.pt"))

    def get_world_space(self):
        radius= {'punch': 1.5, 'die': 1.5, 'holder': 1.5}
        pos_plate = np.load(self.raw_dir+"/pid_81000001_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32)
        pos_punch = np.load(self.raw_dir+ "/pid_11000001_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32)
        pos_die = np.load(self.raw_dir+ "/pid_1_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32)
        pos_holder = np.load(self.raw_dir+ "/pid_21000001_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32)

        pos_punch, pos_holder, pos_die= self.adjust_tool_pos(pos_plate, pos_punch, pos_holder, pos_die)

        world_edges= []
        pos= pos_plate
        node_types= np.ones(len(pos_plate), dtype=np.int64)*NodeType.PLATE

        p_e= kdtree_nearest_graph(pos_plate, pos_punch, radius['punch'])
        p_e[0]+=len(pos)
        world_edges.append(p_e)
        pos= np.vstack((pos, pos_punch))
        node_types= np.hstack((node_types, torch.ones(len(pos_punch), dtype=torch.int64)*NodeType.PUNCH))

        h_e= kdtree_nearest_graph(pos_plate, pos_holder, radius['holder'])
        h_e[0]+=len(pos)
        world_edges.append(h_e)
        pos= np.vstack((pos, pos_holder))
        node_types= np.hstack((node_types, torch.ones(len(pos_holder), dtype=torch.int64)*NodeType.HOLDER))

        d_e= kdtree_nearest_graph(pos_plate, pos_die, radius['die'])
        d_e[0]+=len(pos)
        world_edges.append(d_e)
        pos= np.vstack((pos, pos_die))
        node_types= np.hstack((node_types, torch.ones(len(pos_die), dtype=torch.int64)*NodeType.DIE))

        # world_edges= torch.from_numpy(np.hstack(world_edges))
        pos= torch.from_numpy(pos)
        node_types= torch.from_numpy(node_types)
        return world_edges, pos, node_types

    def adjust_tool_pos(self, pos_plate, pos_punch, pos_holder, pos_die):
        c= pos_plate[:, 2].max()
        pos_punch[:, 2]+= c-pos_punch[:, 2].max()-1
        pos_holder[:, 2]+= c-pos_holder[:, 2].max()-1
        pos_die[:, 2]+= c-pos_die[:, 2].min()+1
        return pos_punch, pos_holder, pos_die
    
    def rel_pos(self, edge_index, pos):
        s, r= edge_index
        temp= pos[s]-pos[r]
        return self.normalize_(pos[s]-pos[r])

    def normalize_(self, x, name=None):
        min, max= x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values
        if name is not None:
            torch.save({"min":min, "max":max}, osp.join(self.processed_dir, rf"{name}_min_max"))
        range= max-min 

        #handling zeroes in range
        constant_mask = range < 10 * torch.finfo(range.dtype).eps
        range[constant_mask] = 1.0
        
        return (x-min)/range

    def len(self):
        return self.data_params.total_size
    
    def get(self, idx):
        data= torch.load(osp.join(self.processed_dir, rf"data_{idx}.pt"))
        return data 

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, r'Daten')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, r'Mesh_GraphNet_Daten_after_preprocessing')   
    
    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return [r"data_0.pt", rf"data_{self.len()-1}.pt"]
