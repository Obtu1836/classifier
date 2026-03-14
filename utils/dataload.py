from config.Config import PathCfg,DatasetCfg
from .process import MakeLoader,train_forms,val_forms

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
