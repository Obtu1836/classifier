import Net
from numpy.typing import NDArray
import torch as th
from torch import nn 
from dataclasses import asdict
from torch.nn import Module
from torch.optim import SGD,Adam,Optimizer
from torch.optim import lr_scheduler
from config.Config import TrainCfg,ModelCfg
from torch.optim.lr_scheduler import LRScheduler
from config.Config import train_cfg

class Factory:

    @staticmethod
    def create_optimizer(net_params, train_cfg: TrainCfg) -> Optimizer:
        name = train_cfg.optimizer_name

        if name == 'sgd':
            sgd=train_cfg.sgd
            return SGD(net_params,**asdict(sgd))
        adam=train_cfg.adam
        return Adam(net_params,**asdict(adam))

    @staticmethod
    def create_model(model_cfg: ModelCfg) -> Module:
        model_cls = getattr(Net, model_cfg.name)
        return model_cls(num_classes=model_cfg.num_classes)

    @staticmethod
    def create_criterion(weight: NDArray | None, train_cfg: TrainCfg):
        device = train_cfg.device
        if weight is not None:
            weights: th.Tensor = th.from_numpy(weight).float().to(device)
            return nn.CrossEntropyLoss(weights, label_smoothing=train_cfg.label_smoothing)
        return nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    
    @staticmethod
    def create_lrsche(lrsche_name:str,optimizer) ->LRScheduler:

        if lrsche_name=='cosin':
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)
        return lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,
                                              patience=10,threshold_mode='abs')

if __name__ == '__main__':
    net=nn.Linear(512,10)
    Factory.create_optimizer(net.parameters(),train_cfg)
