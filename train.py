import Net
from numpy.typing import NDArray
import torch as th
import wandb
from pathlib import Path
from torch import nn
from torch.nn import Module
from rich import print
from rich.progress import track
from Net.unfreeze import create_finetune

from config.Config import (path_cfg, dataset_cfg, model_cfg, train_cfg, process_cfg, finetune_cfg,
                           PathCfg, DatasetCfg, ModelCfg, TrainCfg)

from dataclasses import asdict
from torch.optim import Adam, SGD, Optimizer
from utils.process import MakeLoader, train_forms, val_forms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler


class AnimalDataLoader:

    def __init__(self, mode: str, path_cfg: PathCfg, dataset_cfg: DatasetCfg):

        self.mode = mode
        self.path_cfg = path_cfg
        self.params: dict = dataset_cfg.dataloaderparams
        self.use_sampling_weight: bool = dataset_cfg.use_sampling_weight
        self.use_loss_weight: bool = dataset_cfg.use_loss_weight

    def get_loader(self):

        if self.mode == 'train':
            loader = MakeLoader(self.path_cfg.train_dir, train_forms)
            loader, class_label, loss_weight = loader.make_loader(
                self.use_sampling_weight, self.use_loss_weight, **self.params)

        elif self.mode == 'val':
            loader = MakeLoader(self.path_cfg.val_dir, val_forms)
            loader, class_label, loss_weight = loader.make_loader(
                False, False, **self.params)

        return loader, class_label, loss_weight


class Factory:

    @staticmethod
    def create_optimizer(net_params, train_cfg: TrainCfg) -> Optimizer:
        lr = train_cfg.learning_rate
        name = train_cfg.optimizer

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


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion,
                 lr_sche: LRScheduler,
                 device: str,
                 save_path: Path,
                 model_name: str,
                 normalize: dict):

        self.device = device
        self.model: Module = model.to(self.device)
        self.optimizer: Optimizer = optimizer
        self.lr_sche: LRScheduler = lr_sche
        self.criterion = criterion
        self.save_path = save_path
        self.model_name = model_name
        self.normalize = normalize

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
    def _evaluate_epoch(self, val_load, class_information):
        self.model.eval()
        total_loss, total_num = 0.0, 0.0
        samples_length = len(val_load.dataset)
        example_imgs = []

        for tensor, label in track(val_load, description='evaluating...'):
            tensor, label = tensor.to(self.device), label.to(self.device)
            output = self.model(tensor)
            pred = th.argmax(output, dim=1)

            mis_ind = (pred != label).nonzero().ravel()

            total_num += (pred == label).sum().item()
            loss = self.criterion(output, label)
            total_loss += loss.item() * tensor.size(0)

            if len(mis_ind) == 0:
                continue

            mis_pred = pred[mis_ind].cpu()
            mis_label = label[mis_ind].cpu()
            mis_tensor = tensor[mis_ind].cpu()

            for i in range(len(mis_pred)):
                if len(example_imgs) >= 108:
                    break
                ori_tensor = mis_tensor[i] * self.std + self.mean
                ori_tensor = ori_tensor.clamp(0, 1)
                example_imgs.append(
                    wandb.Image(
                        ori_tensor,
                        caption=f'pred:{class_information[mis_pred[i].item()]} true:{class_information[mis_label[i].item()]}'
                    )
                )

        avg_loss = total_loss / samples_length
        avg_acc = total_num / samples_length

        return avg_loss, avg_acc, example_imgs

    def fit(self, train_loader, val_loader, epochs, class_information,
            resume: bool = True,
            **kwargs):

        start_epoch, init_acc, is_unfrozen, checkpoint = 0, 0.0, False, None

        if resume:
            start_epoch, init_acc, is_unfrozen, checkpoint = self.load_model()

        # wandb: 续训时用固定 id 恢复同一个 run，新训练则创建全新 run
        wandb_id=self.model_name+'-'+self.optimizer.__class__.__name__
        if resume and checkpoint is not None:
            wandb.init(project='animals classifier',
                       id=wandb_id, resume='must',
                       name=wandb_id)
        else:
            wandb.init(project='animals classifier',
                       id=wandb_id, resume=False,
                       name=wandb_id)

        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')

        if not self.save_path.exists():
            self.save_path.mkdir()

        # 1. 创建 finetune 实例
        finetune = create_finetune(
            kwargs.get('strategy'),
            self.model,
            kwargs.get('head_lr', 1e-4),
            kwargs.get('backbone_lr', 1e-5),
            kwargs.get('optim_name', 'adam'),
            epoch=kwargs.get('epoch', 15),
            maxlen=kwargs.get('maxlen', 15)
        )

        # 核心修复：重新定义优化器结构
        if resume and checkpoint is not None:
            if is_unfrozen and finetune is not None:
                print("检测到历史全量微调状态，重构 2 组参数优化器...")
                finetune._unfrozen = True
                finetune._unfreeze_all()          # 关键！先解冻所有参数
                self.optimizer = finetune._rebuild_optim()  # 再重构优化器
            else:
                print("检测到历史单组参数状态，重构单组优化器...")
                # 使用 Factory 重新创建一个干净的、只有 1 组参数的优化器
                self.optimizer = Factory.create_optimizer(
                    self.model.parameters(), train_cfg)

            # 结构现在绝对一致了，可以直接加载
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # 重新关联 scheduler
            self.lr_sche = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2)
            self.lr_sche.load_state_dict(checkpoint['lr_sche'])

        self.mean = th.tensor(self.normalize['mean'])[:, None, None]
        self.std = th.tensor(self.normalize['std'])[:, None, None]

        for i_epoch in range(start_epoch, epochs):

            print(f"第{i_epoch+1}/{epochs}轮训练...")

            train_loss, train_acc = self._train_epoch(train_loader)
            print(f"{"train_loss":<15}: {train_loss:.4f}")
            print(f"{"train_acc":<15}: {train_acc:.4f}")
            wandb.log({'epoch': i_epoch,
                       'train_loss': train_loss,
                       'train_acc': train_acc})

            val_loss, val_acc, wrong_imgs = self._evaluate_epoch(
                val_loader, class_information)
            print(f"{"val_loss":<15}: {val_loss:.4f}")
            print(f"{"val_acc":<15}: {val_acc:.4f}")
            wandb.log({'epoch': i_epoch,
                       'val_loss': val_loss, 'val_acc': val_acc,
                       'mis_img': wrong_imgs})

            init_acc = self.save_model(
                init_acc, val_acc, i_epoch, class_information)

            if finetune is not None:
                # 获取 step 的返回值
                new_optim, new_sche = finetune.step(
                    i_epoch, val_acc, self.optimizer)

                # 如果发生了状态切换（开启了全量微调），更新训练器的成员变量
                if new_sche is not None:
                    self.optimizer = new_optim
                    self.lr_sche = new_sche

            self.lr_sche.step()

            if self.device == 'mps':
                th.mps.empty_cache()
            elif self.device == 'cuda':
                th.cuda.empty_cache()

    def save_model(self, init_acc, current_acc, epoch, class_information):

        if current_acc > init_acc:
            checkpoint = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_sche': self.lr_sche.state_dict(),
                'best_acc': current_acc,
                'class_information': class_information,
                'is_unfrozen': len(self.optimizer.param_groups) > 1
            }
            th.save(checkpoint, self.save_path / f"{self.model_name}_best.pt")
            init_acc = current_acc

        return init_acc

    def load_model(self):

        load_path = self.save_path / f"{self.model_name}_best.pt"
        if not load_path.exists():
            return 0, 0.0, False, None

        checkpoint = th.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        return (checkpoint['epoch'],
                checkpoint['best_acc'],
                checkpoint.get('is_unfrozen', False),
                checkpoint)  # 返回整个 checkpoint 供后续使用


def main():

    train_loader, class_information, loss_weight = AnimalDataLoader(
        'train', path_cfg, dataset_cfg).get_loader()

    val_loader, _, _ = AnimalDataLoader(
        'val', path_cfg, dataset_cfg).get_loader()

    model = Factory.create_model(model_cfg)
    optimizer = Factory.create_optimizer(model.parameters(), train_cfg)
    criterion = Factory.create_criterion(loss_weight, train_cfg)
    lr_sche = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    trainer = Trainer(
        model, optimizer, criterion, lr_sche,
        device=train_cfg.device,
        save_path=path_cfg.checkpoint,
        model_name=model_cfg.name,
        normalize=process_cfg.normalize
    )
    trainer.fit(train_loader,
                val_loader,
                train_cfg.epochs,
                class_information,
                resume=train_cfg.resume,
                optim_name=train_cfg.optimizer,
                **asdict(finetune_cfg))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bold red]训练已手动终止[/bold red]")
    finally:
        wandb.finish()
