import torch 
from torch.utils.data import DataLoader

from trainer import TrainerBase, load_checkpoint
from graph_networks.common import SplitDataset

from graph_networks.parameters import Parameters
from graph_networks import Simple_Dataset
from graph_networks.models import SimpleAnn
from graph_networks.normalizer import Normalizer 

# import sys 
# import graph_networks
# sys.modules['normalizer'] = graph_networks.normalizer

class Trainer(TrainerBase):

    def __init__(self, params: Parameters, model: torch.nn.Module, **kwargs) -> None:
        super().__init__(params, model=model, **kwargs)
    
    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=1, persistent_workers=True)
        return dataloader_

    def train_step(self, data):
        x,y,_ = data[0],data[1],data[2]
        x=x.to(self.params.hyper.device)
        y=y.to(self.params.hyper.device)
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data):
        x,y,_ = data[0],data[1],data[2]
        x=x.to(self.params.hyper.device)
        y=y.to(self.params.hyper.device)
        prediction= self.model(x)
        loss= self.criterion(prediction, y)
        return loss.item()
    
    def test_step(self, data):
        x,y,pos = data[0],data[1],data[2]
        x=x.to(self.params.hyper.device)
        y=y.to(self.params.hyper.device)
        pos= pos.to(self.params.hyper.device)
        prediction= self.model(x)
        loss= self.criterion(prediction, y)
        return loss, prediction
    
    def viz_step(self, idx, y_PRED, data):
        _, y, pos= data
        y= y.to('cpu')
        y_PRED= y_PRED.to('cpu')

        y_PRED = self.output_norm.inverse(y_PRED)
        y = self.output_norm.inverse(y)
        self.viz.plot(idx, y_PRED, y, pos)

if __name__=="__main__":
    params= Parameters()
    params.data.data_dir= r"../../Thesis/04_Kreuznapf_9857_Samples_6_Var"
    # params.hyper.device= 'cuda:0'
    # print(params.data.data_dir.split("_"))
    # exit()
    params.hyper.batch_size= 1
    params.hyper.num_epochs= 1

    dataset= Simple_Dataset(params.data)

    trainset, valset= SplitDataset(data_params=params.data, dataset=dataset, train=True)
    testset, _= SplitDataset(data_params=params.data, dataset=dataset, train=False)
    
    # load_path= r"checkpoints\ckpt_01_21_48\m-SimpleAnn_e-4000_ts-1000_l-0.02.pt"
    # params_ckpt, model_ckpt, trainer_ckpt= load_checkpoint(load_path)

    model= SimpleAnn(5, 50, 4981, 4)
    # model.load_state_dict(model_ckpt)

    output_norm= Normalizer.load_state(params.data.data_dir+r"/Daten/processed/output_norm.pkl")

    trainer= Trainer(params, model)
    # trainer= Trainer.load_from_checkpoint(trainer_ckpt, params, model)
    trainer.train(trainset, valset)
    trainer.test(testset, output_norm)