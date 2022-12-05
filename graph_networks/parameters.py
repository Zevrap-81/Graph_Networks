from dataclasses import dataclass, field
import torch 
import torch.nn as nn

def get_idle_device():
    """Get the most idle device on the network"""
    device_= f"cuda:{0}"
    util= torch.cuda.utilization(device_)
    for i in range(1, torch.cuda.device_count()):
        if torch.cuda.utilization(i) < util:
            device_= f'cuda:{i}'
    
    return device_

@dataclass
class BaseParameters:
    @classmethod
    def load_from_checkpoint(cls, dict):
        return cls(**dict)


    def state_dict(self):
        return self.__dict__

def outliers_list():
    return [179, 481, 686, 4640, 7591, 9598]

@dataclass
class DataParameters(BaseParameters):
    # data_dir: str= r"06_Kreuznapf_1183_Samples_6_Var_time_equidistan"
    data_dir: str= r"04_Kreuznapf_9857_Samples_6_Var"
    ckpt_base_dir:str= r".."
    # outliers_list:list = []
    outliers_list: list= field(default_factory= outliers_list)
    fill_value: int= -1
    split_ratio: float= 0.8
    num_nodes: int= 4981

    n_train: int= 1000

    @property
    def num_samples(self):
        return int(self.data_dir.split("_")[2])

    @property
    def total_size(self):
        return self.num_samples- len(self.outliers_list)

    @property
    def test_size(self):
        return self.total_size-self.n_train

    @property
    def train_size(self):
        return int(self.n_train*self.split_ratio)

    @property
    def val_size(self):
        return self.n_train-self.train_size


def pool_ratios() -> list:
    return [.9, .7, .6, .5]

@dataclass
class Hyperparameters(BaseParameters):

    module: str = None # "GraphUNet_c"     # if you don't want to explicitly provide the model give the name here 
    in_channel: int= 5
    hidden_channel: int= 128
    out_channel: int= 4
    msg_channel: int= 128
    depth: int= 4
    pool_ratios: list = field(default_factory= pool_ratios)
    sum_res: bool= True
    act: nn.ReLU= nn.ReLU()
    message_passing_steps: int=5

    optimizer: str= 'Adam'
    lr: float= 1e-2
    use_reg: bool= True
    use_lrscheduling: bool= True
    load_opt_state: bool = True
    lambda_reg: float= 1e-2
    
    device_: str= None 
    batch_size: int= 75
    num_epochs: int= 5
    ####################################################################

    @property
    def device(self):
        if self.device_ is not None:
            return self.device_
        
        #if gpu not available
        if not torch.cuda.is_available():
            self.device_= 'cpu'
            print(f"Device is set as {self.device_}")
            return self.device_

        #select the idle gpu if more than one available
        else:
            self.device_= get_idle_device()
            print(f"Device is set as {self.device_}")
            return self.device_

    @device.setter
    def device(self, val:str):
        self.first_time= False
        self.device_= val 


@dataclass
class VizParameters(BaseParameters):
    visualize: bool= True
    l: int= 30
    dir: str= "visualisations"
    save: bool= True
    show: bool= False 
    multiple_figs: bool= False
    indices:list= None

@dataclass
class Parameters(BaseParameters):
    data: DataParameters= DataParameters()
    hyper: Hyperparameters= Hyperparameters()
    viz: VizParameters= VizParameters()


    @classmethod
    def load_from_checkpoint(cls, ckpt):
        # https://www.pythontutorial.net/python-oop/python-__new__/#:~:text=Summary-,The%20__new__()%20is%20a%20static%20method%20of%20the%20object,to%20initialize%20the%20object's%20attributes.
        params=  object.__new__(cls)
        params.data= DataParameters.load_from_checkpoint(ckpt["data"])
        params.hyper= Hyperparameters.load_from_checkpoint(ckpt["hyper"])
        params.viz= VizParameters.load_from_checkpoint(ckpt["viz"])
        return params

    def state_dict(self):
        return {"data": self.data.state_dict(), 
                "hyper": self.hyper.state_dict(), 
                "viz": self.viz.state_dict()}

    def save_human_readable(self, path):
        file = path + "/" + "parameters.txt"
        model_name = self.hyper.module
        
        lines = [rf"learning rate : {self.hyper.lr}", 
                 rf"Train Samples(n_train) : {self.data.n_train}",
                 rf"model_name : {model_name}", 
                 rf"batch_size : {self.hyper.batch_size}", 
                 rf"train/val split: {self.data.split_ratio}"]
        with open(file, 'w') as f:
            for line in lines:
                f.writelines(line)
        
if __name__=="__main__":
    params= Parameters()
    ## testing loading and saving functionality
    # params.data.split_ratio= 0.1

    # torch.save({'params': params.state_dict()}, 'params.pt')
    # del params

    # ckpt= torch.load('params.pt')

    # par= Parameters.load_state_dict(ckpt['params'])
    # print(par.data.split_ratio) 