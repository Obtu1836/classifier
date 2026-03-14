import torch as th
import wandb
from pathlib import Path
from torch import nn
from torch.nn import Module
from rich import print
from rich.progress import track
from Net.unfreeze import create_finetune

from config.Config import TrainCfg
from utils.factory import Factory
from utils.dataload import AnimalDataLoader
from dataclasses import asdict
from torch.optim import Optimizer
from torch.optim import lr_scheduler

from torch.optim.lr_scheduler import LRScheduler
from config.Config import path_cfg, dataset_cfg, model_cfg, train_cfg, process_cfg, finetune_cfg



class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion,
                 lr_sche: LRScheduler,
                 device: str,
                 save_path: Path,
                 model_name: str,
                 lrsche_name:str,
                 normalize: dict):

        self.device = device
        self.model: Module = model.to(self.device)
        self.optimizer: Optimizer = optimizer
        self.lr_sche: LRScheduler = lr_sche
        self.criterion = criterion
        self.save_path = save_path
        self.model_name = model_name
        self.lrsche_name=lrsche_name
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
        '''
        常规的验证代码
        '''
        self.model.eval()
        total_loss, total_num = 0.0, 0.0
        samples_length = len(val_load.dataset)#
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

    def fit(self, train_loader, val_loader,  class_information, train_cfg: TrainCfg, **kwargs):
        '''
        1 常规的训练代码
        2 wandb的id 是依据 用来wandb 进行resume的标识 本例中使用模型名称-优化器名称-学习率调度器名称
        作为标识 
        3 涉及多个状态的切换(model,optis,lrsche的交叉选择) 以及resume 
        
        '''
        epochs,resume=train_cfg.epochs,train_cfg.resume
        start_epoch, best_acc, is_unfrozen, checkpoint = 0, 0.0, False, None

        if resume:
            start_epoch, best_acc, is_unfrozen, checkpoint = self.load_model()

        # wandb: 续训时用固定 id 恢复同一个 run，新训练则创建全新 run
        wandb_id=self.model_name+'-'+self.optimizer.__class__.__name__+'-'+self.lr_sche.__class__.__name__
        if resume and checkpoint is not None:
            wandb.init(project='animals classifier',
                       id=wandb_id, resume='must',
                       name=wandb_id)
        else:
            wandb.init(project='animals classifier',
                       id=wandb_id, resume=False, #resume=False 新建项目
                       name=wandb_id)

        wandb.define_metric('epoch')# 定义一个指标
        #*通配符 表示所有的监控指标(loss,acc等)都以epoch为x轴
        # 因为本例中 保存的acc是历史最好的 也就意味着极大的可能 重新继续训练时默认的步进step 会重复
        # 而在保存时epoch 和acc的数量是对应的 所以能够保证图像的连续性
        wandb.define_metric('*', step_metric='epoch')

        if not self.save_path.exists():
            self.save_path.mkdir()

        # 1. 创建 finetune 实例
        '''
        全局微调模块
        使用预训练模型时 开始训练的阶段 只训练分类头 
        然后根据指标(epoch或者acc) 确定何时进入全量参数微调
        epoch 指标 监视epoch 当训练到达epoch时 开始进入全量调参
        acc指标 监视val_acc 内部维护了一个maxlen长度的deque,当deque内部的数据
        稳定后 开始全量调参
        '''
        kwargs.update(asdict(train_cfg)) #需要从train_cfg拿到optimize_name 和lrsche_name 所以合并成一个字典 方便取值
        finetune = create_finetune(
            kwargs.get('strategy'),
            self.model,
            kwargs.get('head_lr', 1e-4),
            kwargs.get('backbone_lr', 1e-5),
            kwargs.get('optimizer_name', 'adam'),
            kwargs.get('lrsche_name','cosin'),#以上参数只能以位置形式传参
            epoch=kwargs.get('epoch', 15), #epoch maxlen 只能以关键字的形式传参
            maxlen=kwargs.get('maxlen', 15)
        )

        '''
        这一部分是 继续训练的逻辑
        finetune 返回的可能是None 代表没有使用全量微雕 或者是FineTuning的对象 代表使用了全量调参
        其中FineTuning的子类(Epoch或Acc)
        '''

        if resume and checkpoint is not None:
            '''
            首先读取的数据中 如果is_unfrozen为True 并且finetune不是None 
            这意味着已经保存过全量调参了 因为 is_unfrozen判断的是优化器参数组>1 
            将finetune实例的_unfrozen属性设为True 这一步 关系到后续finetune.step内部能否正确实现
            然后解冻所有的参数 
            重新构建优化器并赋值

            如果is_unfrozen为False或者finetune为None 
            直接按配置参数重新构建一个优化器
            '''
            if is_unfrozen and finetune is not None:
                print("检测到历史全量微调状态，重构 2 组参数优化器...")
                finetune._unfrozen = True         # 关键 标识
                finetune._unfreeze_all()          # 关键！先解冻所有参数
                self.optimizer = finetune._rebuild_optim()  # 再重构优化器
            else:
                print("检测到历史单组参数状态，重构单组优化器...")
                # 使用 Factory 重新创建一个干净的、只有 1 组参数的优化器
                #如果只有一个参数组 那就按照配置文件重新创建一个新的优化器
                self.optimizer = Factory.create_optimizer(
                    self.model.parameters(), train_cfg)

            #将新创建好的优化器 加载保存的参数
            #重新创建学习率调度器 并加载保存的参数
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_sche=Factory.create_lrsche(self.lrsche_name,self.optimizer)
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

            best_acc = self.save_model(
                best_acc, val_acc, i_epoch, class_information)

            if finetune is not None:
                # 获取 step 的返回值
                '''
                step返回值tuple[Optimizer,None|LRScheduler]
                当config参数不要求使用微调或者当使用微调 指标(epoch|acc)
                没满足时 返回tuple[Optimizer,None]返回传入的优化器 因为
                优化器状态没变 所以学习率调度器也不会变化 所以是None
                当config开启微调且指标满足时 会建立新的优化器和学习率调度器
                返回新的优化器和新的学习率调度器
                如果学习率调度器状态发生变化 (None就是没变化,LRScheduler就说明发生了变化)
                    重新赋值
                '''
                new_optim, new_sche = finetune.step(
                    i_epoch, val_acc, self.optimizer)

                # 如果发生了状态切换（开启了全量微调），更新训练器的成员变量
                if new_sche is not None:
                    self.optimizer = new_optim
                    self.lr_sche = new_sche

            if isinstance(self.lr_sche,lr_scheduler.ReduceLROnPlateau):
                self.lr_sche.step(val_acc)
            else:
                self.lr_sche.step()

            if self.device == 'mps':
                th.mps.empty_cache()
            elif self.device == 'cuda':
                th.cuda.empty_cache()

    def save_model(self, best_acc, current_acc, epoch, class_information):
        '''
        除了常规的保存内容 还需要保存 class_information 类别信息 一方面 测试时直接读取就可以直接用 因为测试部分的代码不打算
        使用dataloader构造 (另一方面 wandb Image也需要用 虽然是直接传参进去的）
        is_unfrozen 参数是 记录使用预训练模型 继续训练时 是否已经进行了全量微调
        判断是否进行了全量微调 只需要判断优化器内部的参数组 如果有2个参数组 说明进行了
        全量微调 2个参数组分别是backbone和head 分别使用不同的学习率
        如果只有一个参数组 那就说明 还没有进行全量微调
        '''
        if current_acc > best_acc:
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
            best_acc = current_acc

        return best_acc

    def load_model(self):

        load_path = self.save_path / f"{self.model_name}_best.pt"
        if not load_path.exists():
            return 0, 0.0, False, None

        checkpoint = th.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        return (checkpoint['epoch'],
                checkpoint['best_acc'],
                checkpoint.get('is_unfrozen', False),#注意保存is_unfrozen
                checkpoint)  # 返回整个 checkpoint 供后续使用


def main():

    train_loader, class_information, loss_weight = AnimalDataLoader(
        'train', path_cfg, dataset_cfg).get_loader()

    val_loader, _, _ = AnimalDataLoader(
        'val', path_cfg, dataset_cfg).get_loader()

    model = Factory.create_model(model_cfg)
    optimizer = Factory.create_optimizer(model.parameters(), train_cfg)
    criterion = Factory.create_criterion(loss_weight, train_cfg)
    lr_sche=Factory.create_lrsche(train_cfg.lrsche_name,optimizer)

    trainer = Trainer(
        model, optimizer, criterion, lr_sche,
        device=train_cfg.device,
        save_path=path_cfg.checkpoint,
        model_name=model_cfg.name,
        lrsche_name=train_cfg.lrsche_name,
        normalize=process_cfg.normalize
    )
   
    trainer.fit(train_loader,
                val_loader,
                class_information,
                train_cfg,
                **asdict(finetune_cfg))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[bold red]训练已手动终止[/bold red]")
    finally:
        wandb.finish()
