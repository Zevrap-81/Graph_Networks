import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from trainer import TrainerBase, load_checkpoint

from graph_networks.common import SplitDataset
from graph_networks.normalizer import Normalizer
from graph_networks.parameters import Parameters
from graph_networks.models import MeshNet
from graph_networks.datageneration import MeshNet_Dataset, Preprocessor

# torch.autograd.set_detect_anomaly(True)

import sys 
import graph_networks
sys.modules['normalizer'] = graph_networks.normalizer

class Trainer(TrainerBase):

    def __init__(self, params: Parameters, model: nn.Module, **kwargs) -> None:
        super().__init__(params, model=model, **kwargs)
    
    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=8, persistent_workers=True)
        return dataloader_

    def loss_config(self):
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.L1Loss()
        def criterion(y_pred, y):
            loss1 = 50*criterion_1(y_pred[..., 0:3], y[..., 0:3]) + 10*criterion_2(y_pred[..., -1], y[..., -1])
            return loss1

        self.criterion = criterion

    def train_step(self, data: Data):
        data = data.to(self.params.hyper.device)
        prediction = self.model(data)
        loss = self.criterion(prediction, data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data:Data):
        data=  data.to(self.params.hyper.device)
        prediction= self.model(data)
        loss= self.criterion(prediction, data.y)
        return loss.item()
    
    def test_loss_config(self):
        self.huberloss = nn.HuberLoss()
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        def test_criterion(pred, target):
            h = self.huberloss(pred, target)
            l1 = self.l1loss(pred, target)
            l2 = self.l2loss(pred, target)
            return {'huber': h.item(),
                    'l1': l1.item(),
                    'l2': l2.item()
                    }

        self.criterion= test_criterion

    def test_step(self, data):
        data=  data.to(self.params.hyper.device)
        prediction= self.model(data)
        loss= self.criterion(prediction, data.y)
        return loss, prediction #no .item() becasue it is taken care of in the criterion

    def viz_step(self, idx, y_PRED, data):
        y_PRED = self.output_norm.inverse(y_PRED)
        y = self.output_norm.inverse(data.y)
        # plot
        self.viz.plot(idx, y_PRED, y, data.pos)


# this 'if' guard is very essential as i am using num_workers>1 in dataloader
if __name__ == "__main__":
    params= Parameters()
    params.data.data_dir= r"../../Thesis/04_Kreuznapf_9857_Samples_6_Var"
    params.hyper.batch_size = 2
    params.hyper.num_epochs= 1

    #Preprocessing
    pre_processor= Preprocessor(params.data) #already saved 
    pre_processor.save_processed_data()

    # load dataset
    dataset= MeshNet_Dataset(params.data)
    trainset, valset, = SplitDataset(dataset, params.data, train=True)
    testset, _ = SplitDataset(dataset, params.data, train=False)

    # load_path= r"checkpoints\ckpt_03_17_37\m-MeshNet_e-4840_ts-1000_l-0.13.pt"
    # params_ckpt, model_ckpt, trainer_ckpt= load_checkpoint(load_path)

    # load model
    model= MeshNet(params)
    # model.load_state_dict(model_ckpt)
    
    #load Normalizer
    output_norm= Normalizer.load_state(rf"{params.data.data_dir}/Daten/processed/output_norm.pkl")
    
    # load trainer
    trainer= Trainer(params=params, model=model)
    # trainer= Trainer.load_from_checkpoint(trainer_ckpt, params, model)
    trainer.train(trainset, valset)     # continue training
    trainer.test(testset, output_norm)               # testing