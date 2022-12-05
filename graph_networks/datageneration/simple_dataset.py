import os.path as osp

import numpy as np
import pandas as pd 

import torch
from torch_geometric.data import Dataset as Dataset_

from tqdm import trange

from graph_networks.parameters import DataParameters
from graph_networks.normalizer import Normalizer

class Simple_Dataset(Dataset_):
    def __init__(self, data_params:DataParameters):
        data_dir= data_params.data_dir
        self.raw_process_dir= data_dir+r"/Daten/processed"
        output_transform= Normalizer.load_state(self.raw_process_dir+r"/output_norm.pkl")
        self.data_params= data_params

        super().__init__(root= data_dir, pre_transform=output_transform)
        
        
        doe_= pd.read_csv(osp.join(data_dir, r"Experiments_1_no_error.csv"), sep=';', decimal=',')
        doe_ = doe_.drop(data_params.outliers_list)
        self.doe_= self.normalize_inputs(doe_)

        # load coords
        self.pos =torch.from_numpy(np.load(data_dir+r'/Daten/pid_81000001_run_1_stage_1.npy')[:, 0:3, 0])

    def process(self):
        for i in trange(self.len(), unit="samples", colour='red'):
            y= torch.from_numpy(np.load(osp.join(self.raw_process_dir, rf"y_{i}.npy")))

            if self.pre_transform is not None:
                y= self.pre_transform.transform_(y)
            torch.save(y, osp.join(self.processed_dir, rf"y_{i}.pt"))

    def get(self, idx):
        y= torch.load(osp.join(self.processed_dir, rf"y_{idx}.pt"))
        return self.doe_[idx], y, self.pos 

    def normalize_inputs(self, doe_):
        doe_= doe_.iloc[:, 1:6].to_numpy() #F_press;fs;h_BT;s_0;sfo;sigy
        doe_= torch.from_numpy(doe_).float()
        min, _= doe_.min(axis=0, keepdims=True)
        max, _= doe_.max(axis=0, keepdims=True)
        np.savez(osp.join(self.processed_dir, r"min_max"), min=min, max=max)
        return (doe_-min)/(max-min)

    def len(self):
        return self.data_params.total_size

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'simple_processed')

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return [r"y_0.pt", rf"y_{self.len()-1}.pt"]
