import os.path as osp
import numpy as np
from scipy.spatial import cKDTree

from random import sample

import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset

from graph_networks.concave_hull import Polygon

def quads_to_edges(connectivity):
  #Not so intuitive but faster code
  i0, i1, i2, i3 = connectivity.T
  ###########################i0<i1####################################
  id_01= i0<i1
  r0=np.where(id_01, i0, i1)
  r1= np.where(~id_01, i0, i1)
  ###########################i1<i2####################################
  id_12= i1<i2
  r0= np.r_[r0, np.where(id_12, i1, i2)]
  r1= np.r_[r1, np.where(~id_12, i1, i2)]
  ###########################i2<i3####################################
  id_23= i2<i3
  r0= np.r_[r0, np.where(id_23, i2, i3)]
  r1= np.r_[r1, np.where(~id_23, i2, i3)]
  ###########################i3<i0####################################
  id_30= i3<i0
  r0= np.r_[r0, np.where(id_30, i3, i0)]
  r1= np.r_[r1, np.where(~id_30, i3, i0)]
  ####################################################################

  edges= np.vstack((r0, r1))
  edges= np.unique(edges, axis=1)

  "because bidirectional"
  edges= np.hstack((edges, np.roll(edges,1, axis=0)))
  return edges



def __quads_to_edges(connectivity):
  edges = set()
  # Fix me for time
  # here an assumption is made that for a quadratic element with [i0, i1, i2, i3] nodes 
  # the edges are e1= [i0, i1], e2 = [i1, i2], e3= [i2, i3], e4= [i3, i0]
  # I have verified this for Kreuznapf01

  for (i0, i1, i2, i3) in connectivity:
      edges.add((i0, i1)) if(i0 < i1) else edges.add((i1, i0))
      edges.add((i1, i2)) if(i1 < i2) else edges.add((i2, i1))
      edges.add((i2, i3)) if(i2 < i3) else edges.add((i3, i2))
      edges.add((i3, i0)) if(i3 < i0) else edges.add((i0, i3))
  edges= sorted(edges)
  edges= np.array(edges)
  
  "because bidirectional"
  return np.vstack((edges, np.roll(edges, 1, axis=1) )).T

def random_sampler(a=0, b=10, l=5):
    r_idx= sample(range(a, b), l)
    r_idx.sort()
    r_idx= iter(r_idx)  

    def next_():
        return next(r_idx, -1)
    return next_ 

# def random_sampler(a=0, b=10, l=5):
#     idx= np.load("i.npy")
#     iter= np.nditer(idx)
#     def next_():
#         return next(iter, np.array(-1)).item()
#     return next_ 

def classify_geometry_types(data_params, dataset):

    output_norm= torch.load(rf"{data_params.data_dir}/Daten_after_preprocessing/output_norm.pt")


    #wrinkled cups 
    raw_dir= osp.join(data_params.data_dir, r"Daten")
    blank = np.load(osp.join(raw_dir, r'pid_81000001_run_1_stage_1.npy'))[:, 0:3, 0]
    blankholder = np.load(osp.join(raw_dir, r'pid_21000001_run_1_stage_1.npy'))[:, 0:3, 0]

    blank[np.abs(blank)<1e-10]= 0.0
    blankholder[np.abs(blankholder)<1e-10]= 0.0
    blankholder_poly= Polygon(blankholder[:, 0:2])
    mask = blankholder_poly.find_mask(blank[:, 0:2])


    #thinned cups
    thickness= []
    z_var= []
    for d in dataset:
        y= d.y 
        y= output_norm.inverse(y)
        thickness.append(y[:, 3])

        masked= d.y[mask, 2]
        z_var.append(masked.var())

    thickness= torch.vstack(thickness)
    thickness= thickness.min(dim=1).values

    z_var= torch.hstack(z_var)

    c= 1.2
    #after visual testing c=2.2, threshold= 0.3074
    thresold= thickness.mean()-c*thickness.std()
    indices_thinned= torch.where(thickness<thresold)[0]


    c= 2.11
    #after visual testing c=2.11, threshold= 0.0062
    wrinkle_thresold= z_var.mean()+c*z_var.std()
    indices_wrinkled= torch.where(z_var>wrinkle_thresold)[0]

    #normal cups 
    idx_all = torch.arange(0, data_params.total_size, dtype=torch.int64)
    indices_perfect = idx_all[(idx_all[:, None] != indices_thinned).all(dim=-1)]
    indices_perfect = indices_perfect[(indices_perfect[:, None] != indices_wrinkled).all(dim=-1)]
    
    file_name= rf"{data_params.data_dir}/indices.pt"
    torch.save({"thinness": (thresold, indices_thinned), "wrinkles": (wrinkle_thresold, indices_wrinkled), "perfect": (0, indices_perfect)}, file_name)

def weighted_random_sampling(data_params, replace=False):
    processed_file_name = rf"{data_params.data_dir}/Daten_indices_{data_params.n_train}.pt"

    if osp.isfile(processed_file_name):
        indices = torch.load(processed_file_name)
        train_indices, val_indices, test_indices = indices['train'], indices['val'], indices['test']
        return train_indices, val_indices, test_indices 
    
    processed_file_name = rf"{data_params.data_dir}/indices.pt"
    if not osp.isfile(processed_file_name):
        raise Exception("Please classify geomtry for your dataset")

    indices_ = torch.load(processed_file_name)
    indices_thinned = indices_['thinness'][1]
    indices_wrinkled = indices_['wrinkles'][1]
    indices_perfect = indices_['perfect'][1]


    N= data_params.total_size
    w_1, w_2, w_3 = N/len(indices_perfect), N/len(indices_thinned), N/len(indices_wrinkled)
    weights= torch.empty(N, dtype=torch.float64)
    weights[indices_perfect] = w_1
    weights[indices_thinned] = w_2
    weights[indices_wrinkled] = w_3

    indices_all = torch.arange(N, dtype=torch.int64)
    train_val_indices= list(WeightedRandomSampler(weights, data_params.n_train, 
                                                  replacement=replace))
    train_val_indices = torch.tensor(train_val_indices, dtype=torch.int64)

    train_indices = train_val_indices[0:data_params.train_size]
    val_indices = train_val_indices[data_params.train_size:]
    test_indices = set_diff_1d(indices_all, train_val_indices, not replace)

    indices = { 'train': train_indices,
                'val': val_indices,
                'test': test_indices}
    torch.save(indices, processed_file_name)
    return train_indices, val_indices, test_indices 


def SplitDataset(dataset, data_params, train=True):
    
    train_indices, val_indices, test_indices= weighted_random_sampling(data_params)

    if train:
        trainset = Subset(dataset, train_indices)
        valset = Subset(dataset, val_indices)
        return trainset, valset
    else:
        testset = Subset(dataset, test_indices)
        valset = Subset(dataset, val_indices)
        return testset, valset



def set_diff_1d(u, v, assume_unique= False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.
    https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
    """
    if not assume_unique:
        u = torch.unique(u)
        v = torch.unique(v)
    return u[(u[:, None] != v).all(dim=1)]


def get_elem_index(elem_conn):
    idxs= []

    for i in range(4):
        n_i= elem_conn.T[i]
        for j in range(4):
            n_j= elem_conn.T[j]
            idxs.append((n_i[:, None] == n_j))
    
    idx= torch.stack(idxs, dim=0).any(dim=0)
    diag= idx.diagonal()
    diag[:]= False 
    idx= torch.vstack(torch.where(idx))
    return idx


def get_elem_index2(elem_conn):
    idx= []
    for i in range(len(elem_conn)):
        node_comp= elem_conn[i]
        for j in range(i+1, len(elem_conn)):
            node_comp2= elem_conn[j]
            if (node_comp[:, None] == node_comp2).any():
                idx.append(torch.tensor([i, j], dtype=torch.long))
    
    idx= torch.vstack(idx).T
    idx= torch.hstack((idx, idx.flipud()))
    return idx


def kdtree_nearest_graph(coord_plate, coord_item, radius=1.5):
  '''
  Creates edges between the nearest neighbours with the given radius
  Coord_plate: float
                coordinated of the plate
  coord_item: float
              coordinates of the external tool in contact
  radius : float
          
  '''
  tree_punch = cKDTree(coord_item)
  index = tree_punch.query_ball_point(coord_plate, radius)

  edges= [np.c_[idx, [i]*len(idx)] for i, idx in enumerate(index) if idx ]
  edges= np.vstack(edges) 
  edges= torch.from_numpy(edges).to(torch.long)
  return edges.T

if __name__ == "__main__":
    a= random_sampler(b= 5400, l= 300)
    print(a())
    print(a())
