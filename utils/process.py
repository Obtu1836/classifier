import numpy as np
from numpy.typing import NDArray
from config.Config import process_cfg, path_cfg, dataset_cfg
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .pretreatment import Resize
from torch.utils.data import DataLoader, sampler


train_forms = transforms.Compose([

    Resize(process_cfg.use_letter, process_cfg.shape, limitbig=True),
    transforms.RandomAffine(20),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomSolarize(threshold=192.0, p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=process_cfg.normalize['mean'],
                         std=process_cfg.normalize['std'])
])

val_forms = transforms.Compose([

    Resize(process_cfg.use_letter, process_cfg.shape, limitbig=True),

    transforms.ToTensor(),
    transforms.Normalize(mean=process_cfg.normalize['mean'],
                         std=process_cfg.normalize['std'])])


class MakeLoader:

    def __init__(self, path, trans):
        self.path = path
        self.trans = trans

    def make_dataset(self):
        dataset = ImageFolder(self.path, self.trans)
        return dataset

    def make_loader(self, sampling_weight=False, loss_weight=False, **kwargs)->\
                                                                tuple[DataLoader,dict,NDArray|None]:
        """
        根据配置创建并返回数据加载器（DataLoader）。

        该方法支持三种模式：
        1. 默认模式：标准数据加载。
        2. 采样权重模式（sampling_weight=True）：解决类别不平衡问题，通过 WeightedRandomSampler 
           使样本数量较少的类在训练中出现的频率更高。
        3. 损失权重模式（loss_weight=True）：计算每个类别的倒数权重，通常用于传递给 
           CrossEntropyLoss 的 weight 参数。

        Args:
            sampling_weight (bool): 是否启用加权随机采样。若为 True，DataLoader 将使用 
                WeightedRandomSampler 且 shuffle 会被强制设为 False。
            loss_weight (bool): 是否计算类别权重。若为 True，将返回用于计算损失函数的类别权重。
            **kwargs: 传递给 torch.utils.data.DataLoader 的额外参数（如 batch_size, num_workers 等）。

        Returns:
            tuple: 包含以下三个元素的元组:
                - loader (DataLoader): 实例化后的 PyTorch 数据加载器。
                - class_idx (dict): 类别映射字典，格式为 {索引: 类别名称}。
                - class_weights (numpy.ndarray or None): 
                    当 loss_weight=True 时，返回 shape 为 (num_classes,) 的权重数组；
                    否则返回 None。

        Note:
            如果同时设置 sampling_weight 和 loss_weight 为 True，函数将优先进入 loss_weight 分支逻辑。
        """
        dataset = self.make_dataset()
        class_idx = dataset.class_to_idx
        class_idx = {k: v for v, k in class_idx.items()}  # 数字 ：label

        if loss_weight:
            targets = np.array(dataset.targets)
            _, class_counts = np.unique(targets, return_counts=True)
            class_weights = 1. / class_counts

            loader = DataLoader(
                dataset, **kwargs, shuffle=True)
            return loader, class_idx, class_weights

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
            return loader, class_idx, None

        else:
            loader = DataLoader(
                dataset, **kwargs, shuffle=True)
            return loader, class_idx, None


if __name__ == '__main__':

    loader = MakeLoader(path_cfg.train_dir, trans=val_forms)
    a, b, c = loader.make_loader(
        dataset_cfg.use_sampling_weight, dataset_cfg.use_loss_weight)
    print(b)
