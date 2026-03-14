import Net
from numpy.typing import NDArray
import torch as th
from torch import nn 
from torch.nn import Module
from torch.optim import SGD,Adam,Optimizer
from torch.optim import lr_scheduler
from config.Config import TrainCfg,ModelCfg
from torch.optim.lr_scheduler import LRScheduler

class Factory:

    @staticmethod
    def create_optimizer(net_params, train_cfg: TrainCfg) -> Optimizer:
        lr = train_cfg.learning_rate
        name = train_cfg.optimizer_name

        if name == 'sgd':
            return SGD(net_params, lr, momentum=0.9, weight_decay=1e-4)
        return Adam(net_params, lr, (0.9, 0.99), weight_decay=5e-4)

    @staticmethod
    def create_model(model_cfg: ModelCfg) -> Module:
        model_cls = getattr(Net, model_cfg.name)
        return model_cls(num_classes=model_cfg.num_classes)

    @staticmethod
    def create_criterion(weight: NDArray | None, train_cfg: TrainCfg):
        device = train_cfg.device
        if weight is not None:
            weights: th.Tensor = th.from_numpy(weight).float().to(device)
            return nn.CrossEntropyLoss(weights, label_smoothing=0.1)
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    
    @staticmethod
    def create_lrsche(lrsche_name:str,optimizer) ->LRScheduler:

        if lrsche_name=='cosin':
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2)
        return lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,
                                              patience=10,threshold_mode='abs')
