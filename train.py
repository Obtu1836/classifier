import Net
import torch as th
import wandb
from torch import nn
from torch.nn import Module
from utils import cfg
from rich import print
from rich.progress import track
from torch.optim import Adam, SGD, Optimizer
from utils.process import MakeLoader, train_forms, val_forms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

# load_dotenv()
# wandb_key=os.getenv('WANDB_API_KEY')

# wandb.login(key=wandb_key)

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
        return model_cls(num_classes=model_cfg['num_classes'])
    
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
        wandb.init(project='animals classifier')
        wandb.config={**cfg['train']}

        wandb.watch(self.model,log='all')

    def _train_epoch(self, loader):

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

        wandb.log({'train_loss':avg_loss,'train_acc':avg_acc})

        print(f"{"average_loss":<15}: {avg_loss:.4f}")
        print(f"{"average_acc":<15}: {avg_acc:.4f}")

    @th.no_grad()
    def _evaluate_epoch(self,model:Module,val_load,criterion,idx):
        model.eval()
        total_loss,total_num=0.0,0.0
        samples_length=len(val_load.dataset)
        example_imgs=[]

        mean=th.tensor(cfg['preprocessing']['normalize']['mean'])[:,None,None].to(self.device)
        std=th.tensor(cfg['preprocessing']['normalize']['std'])[:,None,None].to(self.device)
        for tensor,lable in track(val_load,description='evaluating...'):
            tensor,label=tensor.to(self.device),lable.to(self.device)
            output=model(tensor)
            pred=th.argmax(output,dim=1)
            
            mis_ind=(pred!=label).nonzero().ravel()
            mis_pred=pred[mis_ind]
            mis_label=label[mis_ind]
            mis_tensor=tensor[mis_ind]
            
            total_num+=(pred==label).sum().item()
            loss=criterion(output,label)
            total_loss+=loss.item()*tensor.size(0)
            
            for i in range(len(mis_pred)):
                if len(example_imgs) >= 108:
                    break
                ori_tensor=mis_tensor[i]*std+mean
                ori_tensor=ori_tensor.clamp(0,1)
                example_imgs.append(wandb.Image(ori_tensor,
                                                   caption=f'pred:{idx[mis_pred[i].item()]} true:{idx[mis_label[i].item()]}'))

        
        avg_loss=total_loss/samples_length
        avg_acc=total_num/samples_length

        wandb.log({'val_loss':avg_loss,'val_acc':avg_acc,'mis_img':example_imgs})
        print(f"{"average_loss":<15}: {avg_loss:.4f}")
        print(f"{"average_acc":<15}: {avg_acc:.4f}")

    







    def fit(self, train_loader, val_loader,epochs,idx):

        for epoch in range(epochs):
            print(f"第{epoch+1}/{epochs}轮训练...")
            self._train_epoch(train_loader)
            self._evaluate_epoch(self.model,val_loader,self.criterion,idx)
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
    trainer.fit(train_loader, val_loader,cfg['train']['epochs'],idx)


if __name__ == '__main__':
    main()
