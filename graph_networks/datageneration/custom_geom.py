import enum
import os.path as osp
import numpy as np 
import pandas as pd
import torch
import torch.nn.functional as F 
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.profile import count_parameters

from common import kdtree_nearest_graph, load_checkpoint, quads_to_edges
from concave_hull import Polygon
from normalizer import Normalizer
from di_rough import viz
from graphnet_on_steroids import MeshGraphNet_legacy
from parameters import Parameters
class NodeType(enum.IntEnum):
  PLATE = 0
  PUNCH = 1
  HOLDER = 2
  DIE = 3
  SIZE = 4



class Custom_Dataset(Dataset):
    def __init__(self):
        root= r"02_Custom_01_Samples"
        super().__init__(root)
    
    def process(self):

        connectivity = np.load(osp.join(self.root, r"node_element_id_81000001.npy"))
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
        
        pos_plate = torch.from_numpy(np.load(self.root+"/pid_81000001_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32))
        world_edges= torch.hstack(world_edges)

        i=2
        d= doe_[i]
        e_plate= torch.ones(len(rel_mesh), 2)* torch.unsqueeze(d[[3,4]], 0) #s_0;sfo
        e_plate= torch.hstack((rel_mesh, e_plate))

        e_punch= torch.ones(len(rel_world[0]), 2)* torch.unsqueeze(d[[2,1]], 0) #h_BT;fs
        e_punch= torch.hstack((rel_world[0], e_punch))

        e_die= torch.ones(len(rel_world[2]), 2)* torch.unsqueeze(d[[2,1]], 0) #h_BT;fs
        e_die= torch.hstack((rel_world[2], e_die))

        e_holder= torch.ones(len(rel_world[1]), 2)* torch.unsqueeze(d[[0,1]], 0) #F_press;fs
        e_holder= torch.hstack((rel_world[1], e_holder))
        


        edge_attr= torch.vstack((e_punch, e_die, e_holder))
        u= torch.tensor([[0.4902]], dtype=torch.float32)
        data= Data(x=x, mesh_edge_index=mesh_edges, mesh_edge_attr=e_plate, pos=pos_plate,
                    world_edge_index=world_edges, world_edge_attr=edge_attr, u=u, num_nodes=len(pos))
        
        torch.save(data, osp.join(self.processed_dir, rf"data_{i}.pt"))
    
    def len(self):
        return 1

    def get_world_space(self):
        radius= {'punch': 2, 'die': 4, 'holder': 2}
        pos_plate = np.load(self.root+"/pid_81000001_run_1_stage_1.npy")[:, 0:3, 0].astype(np.float32)
        pos_punch = np.load(self.root+ "/nodes_punch.npy").astype(np.float32)
        pos_die = np.load(self.root+ "/nodes_die.npy").astype(np.float32)
        pos_holder = np.load(self.root+ "/nodes_holder.npy").astype(np.float32)
        final_pos_die= pos_die+np.array([[0.0, 0.0, -54.999992]], dtype=np.float32)
        final_pos_holder= pos_holder+np.array([[0.0, 0.0, -45.596493]], dtype=np.float32)
        pos_punch, pos_holder, pos_die= self.adjust_tool_pos(pos_plate, pos_punch, pos_holder, pos_die)

        pos_punch_2= pos_punch.copy()
        pos_punch_2[pos_punch_2[:, 2]>-16.505, 2]= pos_punch[4, 2]

        pos_die_2= pos_die.copy()
        pos_die_2[pos_die_2[:, 2]<5.5, 2]= -4.5049992

        world_edges= []
        pos= pos_plate
        node_types= np.ones(len(pos_plate), dtype=np.int64)*NodeType.PLATE

        p_e= kdtree_nearest_graph(pos_plate, pos_punch_2, radius['punch'])
        sel_punch= pos_punch[p_e[0]]
        p_e[0]+=len(pos)
        world_edges.append(p_e)
        pos= np.vstack((pos, pos_punch))
        node_types= np.hstack((node_types, torch.ones(len(pos_punch), dtype=torch.int64)*NodeType.PUNCH))

        d_e= kdtree_nearest_graph(pos_plate, pos_die_2, radius['die'])
        sel_die= pos_die[d_e[0]]
        d_e[0]+=len(pos)
        world_edges.append(d_e)
        pos= np.vstack((pos, final_pos_die))
        node_types= np.hstack((node_types, torch.ones(len(pos_die), dtype=torch.int64)*NodeType.DIE))

        h_e= kdtree_nearest_graph(pos_plate, pos_holder, radius['holder'])
        sel_holder= pos_holder[h_e[0]]
        h_e[0]+=len(pos)
        world_edges.append(h_e)
        pos= np.vstack((pos, final_pos_holder))
        node_types= np.hstack((node_types, torch.ones(len(pos_holder), dtype=torch.int64)*NodeType.HOLDER))

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
        return self.normalize_(torch.hstack((temp, torch.norm(temp, dim=-1, keepdim=True))))

    def normalize_(self, x, name=None):
        min, max= x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values
        if name is not None:
            torch.save({"min":min, "max":max}, osp.join(self.processed_dir, rf"{name}_min_max"))
        range= max-min 

        #handling zeroes in range
        constant_mask = range < 10 * torch.finfo(range.dtype).eps
        range[constant_mask] = 1.0
        
        return (x-min)/range

    def get(self, idx):
        data= torch.load(osp.join(self.processed_dir, rf"data_{idx}.pt"))
        return data 

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, r'Daten_after_preprocessing')   
    
    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return [r"data_0.pt"]
    

if __name__=="__main__":
    dataset= Custom_Dataset()
    dataloader = DataLoader(dataset, batch_size=1)
    data= next(iter(dataloader))

    params_ckpt, model_ckpt, trainer_ckpt= load_checkpoint(rf"saved_data\11_21_00_54\checkpoints\m-MeshGraphNet_legacy_e-1000_ts-800_l-0.07.pt")
    params= Parameters.load_from_checkpoint(params_ckpt)

    #Instantiate the model
    model = MeshGraphNet_legacy(params.hyper)
    model.load_state_dict(model_ckpt)
    model.eval()

    print(count_parameters(model))
    # # output_norm= Normalizer.load_state(rf"{params.data.data_dir}/Daten/processed/output_norm.pkl")
    # output_norm= Normalizer.load_state(rf"{params.data.data_dir}/Daten_after_preprocessing/output_norm.pt")
    # input_norm= Normalizer.load_state(rf"{params.data.data_dir}/Daten_after_preprocessing/input_norm.pt")

    # y_pred= model(data).detach()
    # y_pred= output_norm.inverse(y_pred)
    # y_pred= y_pred[:, 0:3]+data.pos

    # y_pred= y_pred.numpy()[:, 0:3]

    # root= r"02_Custom_01_Samples"
    # pos_punch = np.load(root+ "/nodes_punch.npy").astype(np.float32)
    # pos_die = np.load(root+ "/nodes_die.npy").astype(np.float32)
    # pos_holder = np.load(root+ "/nodes_holder.npy").astype(np.float32)
    # final_pos_die= pos_die+np.array([[0.0, 0.0, -54.999992]], dtype=np.float32)

    # viz([y_pred, pos_punch, pos_die, pos_holder])