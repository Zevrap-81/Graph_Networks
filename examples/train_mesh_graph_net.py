import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from trainer import TrainerBase, load_checkpoint

from graph_networks.common import SplitDataset
from graph_networks.parameters import Parameters
from graph_networks.normalizer import Normalizer
from graph_networks.datageneration import Mesh_Dataset
from graph_networks.models import MeshGraphNet_legacy

# import sys 
# import graph_networks
# sys.modules['normalizer'] = graph_networks.normalizer

class Trainer(TrainerBase):
    def __init__(self, params, model:nn.Module, **kwargs) -> None:
        super().__init__(params, model=model, **kwargs)

    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=1, persistent_workers=True)
        return dataloader_

    def loss_config(self):
        criterion_1 = nn.MSELoss()
        criterion_2 = nn.L1Loss()
        def criterion(y_pred, y):
            loss1 = 100*criterion_1(y_pred[..., 0:3], y[..., 0:3]) + 10*criterion_2(y_pred[..., -1], y[..., -1])
            return loss1

        # #this is a custom loss function to penalize wrinkles 
        # mask= torch.load("mask_blankholder.pt")
        # def criterion(y_pred, y):
        #     _y_pred, _y= y_pred.reshape(-1, 4981, 4), y.reshape(-1, 4981, 4)
        #     loss1 = 10*criterion_1(y_pred[..., 0:3], y[..., 0:3]) + 10*criterion_2(y_pred[..., -1], y[..., -1])
        #     loss2= 50*criterion_2(_y_pred[:, mask], _y[:, mask])
        #     return loss1+loss2
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
        # print(y_PRED.isnan().any())
        y = self.output_norm.inverse(data.y)
        # print(y.isnan().any())
        # plot
        self.viz.plot(idx, y_PRED, y, data.pos)

if __name__ == "__main__":
    ##############################
    #    Run model
    ##############################
       
    #Parameters Setup
    params= Parameters()
    params.data.data_dir= r"../../Thesis/04_Kreuznapf_9857_Samples_6_Var"
    params.hyper.batch_size= 2
    params.hyper.num_epochs= 1
    params.hyper.lr= 2e-4

    # params_ckpt, model_ckpt, trainer_ckpt= load_checkpoint("checkpoints\ckpt_11_12_12_46\m-MeshGraphNet_legacy_e-1000_ts-1000_l-0.20.pt")
    # params= Parameters.load_from_checkpoint(params_ckpt)

    #Dataloading
    dataset= Mesh_Dataset(params.data)
    trainset, valset, = SplitDataset(dataset, params.data, train=True)
    testset, _ = SplitDataset(dataset, params.data, train=False)

    #Instantiate the model
    model = MeshGraphNet_legacy(params.hyper)
    # model.load_state_dict(model_ckpt)

    #load Normalizer
    output_norm= Normalizer.load_state(rf"{params.data.data_dir}/Daten/processed/output_norm.pkl")
    
    #Instantiate the trainer
    trainer= Trainer(params, model)
    # trainer= Trainer.load_from_checkpoint(trainer_ckpt, params=params, model=model)

    # training and validation
    trainer.train(trainset, valset)
    #testing and postprocessing
    trainer.test(testset, output_norm)
