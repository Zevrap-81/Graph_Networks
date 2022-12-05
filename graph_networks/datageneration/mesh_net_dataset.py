import os
import os.path as osp

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset

from tqdm import trange, tqdm

from graph_networks.common import get_elem_index, quads_to_edges
from graph_networks.normalizer import Normalizer
from graph_networks.parameters import DataParameters
from graph_networks.concave_hull import Polygon

class Preprocessor:
    def __init__(self, data_params: DataParameters) -> None:
        self.data_params= data_params

        self.data_dir= data_params.data_dir
        self.raw_dir= osp.join(self.data_dir, r"Daten")
        self.process_dir= osp.join(self.raw_dir, r"processed")

        if not osp.exists(self.process_dir):
            os.makedirs(self.process_dir)

        self.num_samples= data_params.num_samples
        self.fill_value= data_params.fill_value

        self.output_norm= Normalizer("output")

        
    def get_inputs(self):
        blank = np.load(osp.join(self.raw_dir, r'pid_81000001_run_1_stage_1.npy'))[:, 0:3, 0]
        stamp = np.load(osp.join(self.raw_dir, r'pid_11000001_run_1_stage_1.npy'))[:, 0:3, 0]
        blankholder = np.load(osp.join(self.raw_dir, r'pid_21000001_run_1_stage_1.npy'))[:, 0:3, 0]

        blank[np.abs(blank)<1e-10]= 0.0
        stamp[np.abs(stamp)<1e-10]= 0.0
        blankholder[np.abs(blankholder)<1e-10]= 0.0


        stamp_poly= Polygon(stamp[:, 0:2])
        blankholder_poly= Polygon(blankholder[:, 0:2])

        masks = np.empty((blank.shape[0],2), dtype=bool)
        masks[:,0] = stamp_poly.find_mask(blank[:, 0:2])
        masks[:,1] = blankholder_poly.find_mask(blank[:, 0:2])

        np.save(osp.join(self.raw_dir, r'masks'), masks)
        masks = np.load(osp.join(self.raw_dir, r'masks.npy'))

        doe_= pd.read_csv(osp.join(self.data_dir, r"Experiments_1_no_error.csv"), sep=';', decimal=',')
        doe_= doe_.iloc[:, 1:6].to_numpy().astype(np.float32) #F_press;fs;h_BT;s_0;sfo;sigy
        doe_= self.transform_inputs(doe_)

        num_nodes= blank.shape[0]
        for i in range(self.num_samples):
            x= np.ones((num_nodes, 5), dtype=np.float32)*self.fill_value
            x[:, [0,1]]= doe_[i, [3,4]]  #blank are for s_0
            x[masks[:,0], 2]= doe_[i, 2] #stamp area for h_BT
            x[masks[:,1], 3]= doe_[i, 0] #Blankholder area for F_press
            x[masks.any(axis=1), 4]= doe_[i, 1]
            yield x 

    
    def get_outputs(self):
        for i in range(self.num_samples):
            sample = np.load(osp.join(self.raw_dir, rf'pid_81000001_run_{i+1}_stage_1.npy')).astype(np.float32)
            sample = sample[:,[6,7,8,12], 1] 
            yield sample  

   
    def filter_outliers(self):
        outliers_list= self.data_params.outliers_list
        for i, (x,y) in enumerate(zip(self.get_inputs(), self.get_outputs())):
            if not i in outliers_list:
                yield x, y

    def save_processed_data(self):

        #if the files already exist then return
        if self.files_exist():
            return

        print("Preprocessing...")
        with tqdm(enumerate(self.filter_outliers()), unit='samples', total=self.num_samples, colour= 'red') as tdata:
            for i, (x,y) in tdata:
                self.output_norm.fit(y)
                #save inputs and outputs
                np.save(osp.join(self.process_dir, rf"x_{i}.npy"), x)
                np.save(osp.join(self.process_dir, rf"y_{i}.npy"), y)

        #save output norm state 
        self.output_norm.save_state(self.process_dir)

        #save edge_data
        connectivity = np.load(osp.join(self.raw_dir, r"node_element_id_81000001.npy"))
        connectivity-= connectivity.min()
        edges= quads_to_edges(connectivity)
        np.save(osp.join(self.process_dir, r"edges.npy"), edges)
        
        print("Done")

    
    def transform_inputs(self, doe_):
        min, max= doe_.min(axis=0, keepdims=True), doe_.max(axis=0, keepdims=True)
        np.savez(osp.join(self.process_dir, r"min_max"), min=min, max=max)
        return (doe_-min)/(max-min)
    
    def files_exist(self):
        f_paths= [
                osp.join(self.process_dir, r"x_0.npy"),
                osp.join(self.process_dir, rf"x_{self.data_params.total_size-1}.npy"),
                osp.join(self.process_dir, r"min_max.npz"),
                osp.join(self.process_dir, r"edges.npy"),
                osp.join(self.process_dir, r"output_norm.pkl")
                ]

        return all([osp.exists(f_path) for f_path in f_paths])


class _Data(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['x','y','pos']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class MeshNet_Dataset(Dataset):
    def __init__(self, data_params: DataParameters=None):
        self.data_params= data_params
        self.data_dir= self.data_params.data_dir

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        output_transform= Normalizer.load_state(osp.join(self.raw_process_dir, r"output_norm.pkl"))

        super().__init__(root= self.data_dir, pre_transform= output_transform)
    
    def process(self):
        num_samples= self.data_params.total_size
        # #load edges
        # edge_index= torch.from_numpy(np.load(osp.join(self.raw_process_dir, r"edges.npy")))
        
        elem_conn = torch.from_numpy(np.load(osp.join(self.raw_dir, r"node_element_id_81000001.npy")))
        elem_conn-= elem_conn.min()

        elem_index = get_elem_index(elem_conn)
        num_elements= elem_conn.size(0)

        # load coords
        pos =torch.from_numpy(np.load(osp.join(self.raw_dir, r'pid_81000001_run_1_stage_1.npy'))[:, 0:3, 0].astype(np.float32))

        # # edgeattributes
        # edge_weight= torch.norm(pos[edge_index[1]]-pos[edge_index[0]], dim=1)

        for i in trange(num_samples, unit="samples", colour='red'):
            x= torch.from_numpy(np.load(osp.join(self.raw_process_dir, rf"x_{i}.npy"))) #, allow_pickle=True
            y= torch.from_numpy(np.load(osp.join(self.raw_process_dir, rf"y_{i}.npy"))) #, allow_pickle=True
            data= Data(x=x, y=y, elem_index=elem_index, pos=pos, num_nodes=num_elements) # ,edge_index=edge_index, edge_weight=edge_weight, num_nodes=self.data_params.num_nodes)

            if self.pre_transform is not None:
                data= self.pre_transform(data)
                
            torch.save(data, osp.join(self.processed_dir, rf"data_{i}.pt"))
    
    def len(self):
        return self.data_params.total_size
    
    def get(self, idx):
        data= torch.load(osp.join(self.processed_dir, rf"data_{idx}.pt"))
        return data 

    @property
    def raw_dir(self) -> str:
        return osp.join(self.data_dir, r'Daten')

    @property
    def raw_process_dir(self) -> str:
        return osp.join(self.raw_dir, r'processed')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.data_dir, r'Daten_after_preprocessing')   
    
    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return [r"data_0.pt", rf"data_{self.len()-1}.pt"]


if __name__ == "__main__":

    data_params= DataParameters()
    data_params.data_dir= r"../../Thesis/04_Kreuznapf_9857_Samples_6_Var"
    
    #Preprocessing
    pre_processor= Preprocessor(data_params) #already saved 
    pre_processor.save_processed_data()

    dataset= MeshNet_Dataset(data_params)
    print(dataset[0].x)
