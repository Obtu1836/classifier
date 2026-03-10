import Net
import torch as th
from torch import nn
from torch.nn import Module
from utils import cfg
from rich import print
from rich.progress import track
from torch.optim import Adam, SGD, Optimizer
from utils.process import MakeLoader, train_forms, val_forms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler


class AnimalDataLoader:

    def __init__(self, mode, path_cfg, dataset_cfg):
        self.mode = mode
        self.path_cfg = path_cfg
        self.params = dataset_cfg['dataloader']
        self.sampling_weight = dataset_cfg['sampling_weight']
        self.loss_weight=dataset_cfg['loss_weight']

    def get_loader(self):

        if self.mode == 'train':
            loader_obj = MakeLoader(self.path_cfg['train_dir'], train_forms)
            loader, class_label,loss_weight = loader_obj.make_loader(
                self.sampling_weight,self.loss_weight, **self.params)
            
        elif self.mode == 'val':
            loader_obj = MakeLoader(self.path_cfg['val_dir'], val_forms)
            loader, class_label,loss_weight = loader_obj.make_loader(False,False, **self.params)

        else:
            loader_obj = MakeLoader(self.path_cfg['test_dir'], val_forms)
            loader, class_label,loss_weight = loader_obj.make_loader(False,False, **self.params)

        return loader, class_label,loss_weight


class Factory:

    @staticmethod
    def create_optimizer(net_params, train_cfg) -> Optimizer:
        lr = train_cfg['learning_rate']
        name = str(train_cfg.get('optimizer', 'adam')).lower()

        if name == 'sgd':
            return SGD(net_params, lr, momentum=0.9, weight_decay=1e-4)
        return Adam(net_params, lr, (0.9, 0.99))

    @staticmethod
    def create_model(model_cfg) -> Module:
        model_cls = getattr(Net, model_cfg['name'])
        return model_cls(model_cfg['num_classes'])
    
    @staticmethod
    def createa_criterion(weight,cfg_train):
        device=cfg_train['device']
        if weight is not None:
            weight=th.from_numpy(weight).float().to(device)
            return nn.CrossEntropyLoss(weight)
        return nn.CrossEntropyLoss()


class Trainer:
    def __init__(self, model, optimizer, criterion, lr_sche, device='cpu'):

        self.model = model.to(device)
        self.optimizer: Optimizer = optimizer
        self.lr_sche: LRScheduler = lr_sche
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):

        self.model.train()
        total_loss, total_num = 0.0, 0.0
        samples_length = len(loader.dataset)

        for tensor, label in track(loader, description='training'):
            tensor, label = tensor.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(tensor)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()*tensor.size(0)
            pred_label = th.argmax(output, dim=1)
            total_num += (pred_label == label).sum().item()

        avg_loss = total_loss/samples_length
        avg_acc = total_num/samples_length

        print(f"{"average_loss":<15}: {avg_loss:.4f}")
        print(f"{"average_acc":<15}: {avg_acc:.4f}")

    @th.no_grad()
    def evaluate_epoch(self,model:Module,val_load,criterion):
        model.eval()
        total_loss,total_num=0.0,0.0
        samples_length=len(val_load.dataset)
        for tensor,lable in track(val_load,description='evaluating...'):
            tensor,label=tensor.to(self.device),lable.to(self.device)
            output=model(tensor)
            pred=th.argmax(output,dim=1)
            total_num+=(pred==label).sum().item()
            loss=criterion(output,label)
            total_loss+=loss.item()*tensor.size(0)
        
        avg_loss=total_loss/samples_length
        avg_acc=total_num/samples_length

        print(f"{"average_loss":<15}: {avg_loss:.4f}")
        print(f"{"average_acc":<15}: {avg_acc:.4f}")

    def fit(self, train_loader, val_loader,epochs):

        for epoch in range(epochs):
            print(f"第{epoch+1}/{epochs}轮训练...")
            self.train_epoch(train_loader)
            self.evaluate_epoch(self.model,val_loader,self.criterion)
            self.lr_sche.step()
            if self.device=='mps':
                th.mps.empty_cache()
            elif self.device =='cuda':
                th.cuda.empty_cache()


def main():

    train_loader, idx,loss_weight = AnimalDataLoader(
        'train', cfg['paths'], cfg['dataset']).get_loader()
    
    val_loader,_,_=AnimalDataLoader('val',cfg['paths'],cfg['dataset']).get_loader()
    
    model = Factory.create_model(cfg['model'])
    optimizer = Factory.create_optimizer(model.parameters(), cfg['train'])
    criterion=Factory.createa_criterion(loss_weight,cfg['train'])

    lr_sche = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    device = cfg['train']['device']
    trainer = Trainer(model, optimizer, criterion, lr_sche, device)
    trainer.fit(train_loader, val_loader,cfg['train']['epochs'])


if __name__ == '__main__':
    main()
