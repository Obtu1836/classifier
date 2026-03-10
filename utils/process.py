import torch as th
import numpy as np
from . import cfg
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .pretreatment import Resize
from torch.utils.data import DataLoader, sampler


procssing=cfg['preprocessing']
train_forms = transforms.Compose([

    Resize(procssing['use_letter'],procssing['shape'], limitbig=True),
    transforms.RandomAffine(20),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomSolarize(threshold=192.0, p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=procssing['normalize']['mean'],
                         std=procssing['normalize']['std'])
])

val_forms=transforms.Compose([

    Resize(procssing['use_letter'],procssing['shape'], limitbig=True),

    transforms.ToTensor(),
    transforms.Normalize(mean=procssing['normalize']['mean'],
                         std=procssing['normalize']['std'])])

class MakeLoader:

    def __init__(self, path, trans):
        self.path = path
        self.trans = trans

    def make_dataset(self):
        dataset = ImageFolder(self.path, self.trans)
        return dataset

    def make_loader(self, sampling_weight=False,loss_weight=False, **kwargs):
        dataset = self.make_dataset()
        class_idx = dataset.class_to_idx

        if loss_weight:
            targets = np.array(dataset.targets)
            _, class_counts = np.unique(targets, return_counts=True)
            class_weights = 1. / class_counts

            loader = DataLoader(
                dataset, **kwargs, shuffle=True)
            return loader, class_idx,class_weights


        if sampling_weight:
            targets = np.array(dataset.targets)
            _, class_counts = np.unique(targets, return_counts=True)
            class_weights = 1. / class_counts
            sample_weights = class_weights[targets]

            # num_samples: 每一轮采样的总数，通常等于 len(dataset)
            # replacement=True: 允许重复采样（权重大的样本会被多次选中）
            samp = sampler.WeightedRandomSampler(
                weights=sample_weights,  # type: ignore
                num_samples=len(dataset),
                replacement=True)

            # 使用了 sampler 后，DataLoader 的 shuffle 必须设为 False (或不设置)
            loader = DataLoader(
                dataset,
                **kwargs,
                sampler=samp, shuffle=False)
            return loader, class_idx,None
        
        else:
            loader = DataLoader(
                dataset, **kwargs, shuffle=True)
            return loader, class_idx,None


if __name__ == '__main__':

    path=cfg['paths']

    loader=MakeLoader(path['train_dir'],trans=val_forms)
    a,b,c=loader.make_loader(cfg['dataset']['sampling_weigh'],cfg['dataset']['loss_weight']**cfg['dataset']['dataloader'])
    # dataset=loader.make_dataset()




