import Net
import torch as th
import wandb
from pathlib import Path
from torch import nn
from torch.nn import Module
from utils import cfg
from rich import print
from rich.progress import track
from torch.optim import Adam, SGD, Optimizer
from utils.process import MakeLoader, train_forms, val_forms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler

MEAN = th.tensor(cfg['preprocessing']['normalize']['mean'])[
    :, None, None]
STD = th.tensor(cfg['preprocessing']['normalize']['std'])[
    :, None, None]


class AnimalDataLoader:

    def __init__(self, mode, path_cfg, dataset_cfg):
        self.mode = mode
        self.path_cfg = path_cfg
        self.params = dataset_cfg['dataloader']
        self.sampling_weight = dataset_cfg['sampling_weight']
        self.loss_weight = dataset_cfg['loss_weight']

    def get_loader(self):

        if self.mode == 'train':
            loader_obj = MakeLoader(self.path_cfg['train_dir'], train_forms)
            loader, class_label, loss_weight = loader_obj.make_loader(
                self.sampling_weight, self.loss_weight, **self.params)

        elif self.mode == 'val':
            loader_obj = MakeLoader(self.path_cfg['val_dir'], val_forms)
            loader, class_label, loss_weight = loader_obj.make_loader(
                False, False, **self.params)

        else:
            loader_obj = MakeLoader(self.path_cfg['test_dir'], val_forms)
            loader, class_label, loss_weight = loader_obj.make_loader(
                False, False, **self.params)

        return loader, class_label, loss_weight


class Factory:

    @staticmethod
    def create_optimizer(net_params, train_cfg) -> Optimizer:
        lr = train_cfg['learning_rate']
        name = str(train_cfg.get('optimizer', 'adam')).lower()

        if name == 'sgd':
            return SGD(net_params, lr, momentum=0.9, weight_decay=1e-4)
        return Adam(net_params, lr, (0.9, 0.99), weight_decay=5e-4)

    @staticmethod
    def create_model(model_cfg) -> Module:
        model_cls = getattr(Net, model_cfg['name'])
        return model_cls(num_classes=model_cfg['num_classes'])

    @staticmethod
    def createa_criterion(weight, cfg_train):
        device = cfg_train['device']
        if weight is not None:
            weight = th.from_numpy(weight).float().to(device)
            return nn.CrossEntropyLoss(weight, label_smoothing=0.1)
        return nn.CrossEntropyLoss(label_smoothing=0.1)


class Trainer:
    def __init__(self, model, optimizer, criterion, lr_sche):

        self.device = 'mps' if th.backends.mps.is_available() else 'cuda'
        self.model: Module = model.to(self.device)
        self.optimizer: Optimizer = optimizer
        self.lr_sche: LRScheduler = lr_sche
        self.criterion = criterion
        self.save_path = Path(cfg['paths']['checkpoint_dir'])
        wandb.init(project='animals classifier')
        wandb.config = {**cfg['train']}

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

        return avg_loss, avg_acc

    @th.no_grad()
    def _evaluate_epoch(self, model: Module, val_load, criterion, idx):
        model.eval()
        total_loss, total_num = 0.0, 0.0
        samples_length = len(val_load.dataset)
        example_imgs = []

        
        for tensor, lable in track(val_load, description='evaluating...'):
            tensor, label = tensor.to(self.device), lable.to(self.device)
            output = model(tensor)
            pred = th.argmax(output, dim=1)

            mis_ind = (pred != label).nonzero().ravel()
            mis_pred = pred[mis_ind]
            mis_label = label[mis_ind]
            mis_tensor = tensor[mis_ind]

            total_num += (pred == label).sum().item()
            loss = criterion(output, label)
            total_loss += loss.item()*tensor.size(0)

            for i in range(len(mis_pred)):
                if len(example_imgs) >= 108:
                    break
                ori_tensor = mis_tensor[i]*STD+MEAN
                ori_tensor = ori_tensor.clamp(0, 1)
                example_imgs.append(wandb.Image(ori_tensor,
                                                caption=f'pred:{idx[mis_pred[i].item()]} true:{idx[mis_label[i].item()]}'))

        avg_loss = total_loss/samples_length
        avg_acc = total_num/samples_length

        return avg_loss, avg_acc, example_imgs

    def fit(self, train_loader, val_loader, epochs, idx,resume=False):

        start_epoch,init_acc=0,0.0
        if resume:
            start_epoch, init_acc = self.load_model()

        if not self.save_path.exists():
            self.save_path.mkdir()

        for epoch in range(start_epoch, epochs):
            print(f"第{epoch+1}/{epochs}轮训练...")
            train_loss, train_acc = self._train_epoch(train_loader)
            print(f"{"train_loss":<15}: {train_loss:.4f}")
            print(f"{"train_acc":<15}: {train_acc:.4f}")
            wandb.log({'train_loss': train_loss,
                      'train_acc': train_acc}, step=epoch)

            val_loss, val_acc, wrong_imgs = self._evaluate_epoch(
                self.model, val_loader, self.criterion, idx)
            print(f"{"val_loss":<15}: {val_loss:.4f}")
            print(f"{"val_acc":<15}: {val_acc:.4f}")
            wandb.log({'val_loss': val_loss, 'val_acc': val_acc,
                      'mis_img': wrong_imgs}, step=epoch)

            init_acc = self.save_model(init_acc, val_acc, epoch,idx)

            self.lr_sche.step()
            if self.device == 'mps':
                th.mps.empty_cache()
            elif self.device == 'cuda':
                th.cuda.empty_cache()

    def save_model(self, best_acc, current_acc, epoch,idx):
        '''
        我的保存逻辑 只保留历史最高分 继续训练也是在最高分的基础上
        '''
        if current_acc > best_acc:
            checkpoint = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_sche': self.lr_sche.state_dict(),
                'best_acc': best_acc,
                'classic_infomation':idx}
            th.save(checkpoint, self.save_path /
                    f"{cfg['model']['name']}_best.pt")

            best_acc = current_acc
        return best_acc

    def load_model(self):

        load_path = self.save_path / f"{cfg['model']['name']}_best.pt"
        if load_path.exists():
            checkpoint = th.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_sche.load_state_dict(checkpoint['lr_sche'])
            return checkpoint['epoch'] + 1, checkpoint['best_acc']
        else:
            return 0, 0.0


def main():

    train_loader, idx, loss_weight = AnimalDataLoader(
        'train', cfg['paths'], cfg['dataset']).get_loader()

    val_loader, _, _ = AnimalDataLoader(
        'val', cfg['paths'], cfg['dataset']).get_loader()

    model = Factory.create_model(cfg['model'])
    optimizer = Factory.create_optimizer(model.parameters(), cfg['train'])
    criterion = Factory.createa_criterion(loss_weight, cfg['train'])
    lr_sche = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    trainer = Trainer(model, optimizer, criterion, lr_sche)
    trainer.fit(train_loader, val_loader, cfg['train']['epochs'], idx)


if __name__ == '__main__':
    main()
