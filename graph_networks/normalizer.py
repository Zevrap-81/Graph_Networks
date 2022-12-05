import numpy as np
import pickle
import torch
from torch_geometric.data import Data
import os.path as osp

class Normalizer: 
    def __init__(self, name:str):
        self.name= name
        self.min_, self.max_= None, None
        self.range_= None
        self.samples_seen= 0
    
    def __call__(self, X:Data):
        return self.transform(X)

    def fit(self, X:np.array):
        min, max= X.min(axis=0, keepdims= True), X.max(axis=0, keepdims= True)
        if self.samples_seen == 0: 
            #First pass 
            self.min_, self.max_= min, max

        else: 
            self.min_, self.max_= np.minimum(min, self.min_), np.maximum(max, self.max_)
        
        self.range_= self.max_-self.min_ 
        self.samples_seen+=1 
    
    def transform(self, X: Data):
        min, max= torch.from_numpy(self.min_), torch.from_numpy(self.max_)
        range= torch.from_numpy(self.range_)

        X.y= (X.y-min)/(range)

        # min, max= X.edge_weight.min(), X.edge_weight.max()
        # X.edge_weight= (X.edge_weight-min)/(max-min)
        
        return X

    def transform_(self, y):
        min, max= torch.from_numpy(self.min_), torch.from_numpy(self.max_)
        range= torch.from_numpy(self.range_)

        y= (y-min)/(range)
        return y
        
    def inverse(self, y):
        device= y.device
        y= y * torch.from_numpy(self.range_).to(device) + torch.from_numpy(self.min_).to(device)
        return y

    def save_state(self, dir:str= ""):
        dir= osp.join(dir,self.name+"_norm.pkl")
        with open(dir, "wb") as pickle_file:
            pickle.dump(self, pickle_file)
    
    @classmethod
    def load_state(cls, dir:str= ""): #todo give an appropriate def val
        with open(dir, "rb") as pickle_file:
            return pickle.load(pickle_file)
