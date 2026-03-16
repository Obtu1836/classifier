from torch.nn import Module
from abc import ABC, abstractmethod
import torch as th
from torch import nn
from collections import deque
from utils.factory import Factory
from torch.optim import Optimizer
from loguru import logger
from torch.optim.lr_scheduler import LRScheduler

ALL_STRATEGY: dict[str, type["FineTuning"]] = {}


def auto_add_strategy(cls):
    name = str.lower(cls.__name__)
    ALL_STRATEGY[name] = cls
    return cls


class FineTuning(ABC):

    def __init__(self, net: Module,
                 model_name:str,
                 head_lr: float,
                 backbone_lr: float,
                 optim_name: str,
                 lrsche_name: str):

        self.net = net
        self.model_name=model_name
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self._unfrozen = False
        self.optim_name = optim_name
        self.lrsche_name = lrsche_name

    @abstractmethod
    def step(self, epoch, acc, optimizer) -> tuple[Optimizer,LRScheduler|None]:
        ...

    def _do_something(self):

        self._unfreeze_all()
        optimizer = self._rebuild_optim()#重建优化器 
        sche = Factory.create_lrsche(self.lrsche_name, optimizer)#重建学习调度器

        self._unfrozen = True#标识位 并将状态设置为 True 防止在训练循环时 重复执行新建等
        opt_name = optimizer.__class__.__name__
        sche_name = sche.__class__.__name__
        logger.success(f'{self.model_name}-{opt_name}-{sche_name[:7]}启用全量微调')

        return optimizer, sche

    def _unfreeze_all(self):# 将所有参数置为求导状态
        for param in self.net.parameters():
            param.requires_grad = True

    def _rebuild_optim(self):
        
        '''
        通过获取分类头的层的名称 将 整个网络的参数分为2部分 分别是主干层和分类层
        因为是在预训练的基础上 所以 需要重新建立优化器 分别对主干层和分类层设置不同
        的较小的学习率
        '''
        head_name = self._get_head_name(self.net)#获取分类层

        backbone_params = [p for n, p in self.net.named_parameters()#筛选主干层参数
                           if head_name not in n and p.requires_grad]

        head_params = [p for n, p in self.net.named_parameters()#筛选分类层参数
                       if head_name in n and p.requires_grad]
        
        params = [{'params': backbone_params, 'lr': self.backbone_lr},
                  {'params': head_params, 'lr': self.head_lr}]

        if self.optim_name == 'sgd': #重建优化器
            return th.optim.SGD(params, weight_decay=5e-4)
        return th.optim.Adam(params, weight_decay=5e-5)

    def _get_head_name(self, net: Module):

        for name, _ in net.named_children():#children 可获取整层 通过遍历 获取最后的一整个分类头
            pass
        return name

'''
下面两个 是启用全量微调的 条件 分别是通过epoch 和acc

通过epoch 的方式是 设置 假如epoch=10 代表 当训练到第10个epoch时进入全量微调
通过acc的方式是 维护一个maxlen的deque 当每一次循环时加入新的acc 当队列的均值
等于新的acc时 说明趋于稳定 此时进入微调 
'''

@auto_add_strategy
class Epoch(FineTuning):

    def __init__(self, *args, epoch: int = 15):
        super().__init__(*args)

        self.epoch = epoch

    def step(self, epoch, acc, optimizer):
        if not self._unfrozen and epoch >= self.epoch:
            return self._do_something()
        return optimizer, None


@auto_add_strategy
class Acc(FineTuning):

    def __init__(self, *args, maxlen=15):
        super().__init__(*args)

        self.acc_deque = deque(maxlen=maxlen)

    def step(self, epoch: int, acc: float, optimizer):

        self.acc_deque.append(acc)
        m = self.acc_deque.maxlen
        if not self._unfrozen and len(self.acc_deque) >= m:  # type: ignore
            avg = sum(self.acc_deque)/len(self.acc_deque)
            if abs(avg-acc) < 1e-3:
                return self._do_something()
        return optimizer, None


def create_finetune(strategy: str | None, net: Module, model_name: str, head_lr,
                    backbone_lr, optimizer_name: str, lrsche_name: str, /, **kwargs)->None|FineTuning:

    import inspect

    if strategy is None:
        return None
    
    
    if "Pretrained" not in model_name:
        logger.error('检测到 非预训练模型 使用了 全量微调 建议修改 strategy为 None或者不写')
        return None

    cls = ALL_STRATEGY[strategy]

    params = inspect.signature(cls.__init__)#获取目标类的初始化参数
    param_map = params.parameters

    filtered = {k: kwargs[k] for k in param_map if k in kwargs}
    return cls(net, model_name,head_lr, backbone_lr, optimizer_name, lrsche_name, **filtered)


if __name__ == '__main__':

    net = nn.Linear(512, 10)
    test_epoch = create_finetune(
        'epoch', net, 'Pretrainedresnet', 1e-4, 1e-5, 'sgd', 'cosin', maxlen=15, s=20, epoch=15)
    test_acc = create_finetune(
        'acc', net, 'Pretrainedresnet', 1e-4, 1e-5, 'sgd', 'reduce', maxlen=15, s=20, epoch=15)

    print(test_epoch.__dict__)
    print(test_acc.__dict__)
