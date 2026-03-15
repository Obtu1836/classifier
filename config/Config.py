from dataclasses import dataclass, field, asdict
import yaml
from pathlib import Path
import torch as th

yaml_path = Path(__file__).resolve().parent.parent/'config.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


@dataclass(frozen=True)
class PathCfg:
    train_dir: Path
    val_dir: Path
    checkpoint: Path


@dataclass(frozen=True)
class DatasetCfg:
    dataloaderparams: dict[str, int] = field(default_factory=dict)
    use_sampling_weight: bool = False
    use_loss_weight: bool = False


@dataclass(frozen=True)
class Preprocessing:
    shape: list[int] = field(default_factory=lambda: [224, 224])
    use_letter: bool = True
    normalize: dict[str, list] = field(
        default_factory=lambda: {'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225]})


def get_device(mode: str) -> str:
    if mode == 'cpu':
        return 'cpu'

    if mode == 'mps':
        if th.backends.mps.is_available():
            return 'mps'
        return 'cuda' if th.cuda.is_available() else 'cpu'

    if th.cuda.is_available():
        return 'cuda'
    if th.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@dataclass(frozen=True)
class TrainCfg:
    epochs: int
    learning_rate: float
    optimizer_name: str
    device: str
    resume: bool
    lrsche_name: str

    def __post_init__(self):
        final_device = get_device(self.device)
        object.__setattr__(self, 'device', final_device)


@dataclass(frozen=True)
class ModelCfg:
    name: str
    num_classes: int


@dataclass(frozen=True)
class FinetuneCfg:
    strategy: str | None
    head_lr: float
    backbone_lr: float
    maxlen: int         # 统一命名
    epoch: int          # 统一命名


path_cfg = PathCfg(Path(cfg['paths']['train_dir']),
                   Path(cfg['paths']['val_dir']),
                   Path(cfg['paths']['checkpoint_dir']))

dataset_cfg = DatasetCfg(cfg['dataset']['dataloaderparams'],
                         cfg['dataset']['use_sampling_weight'],
                         cfg['dataset']['use_loss_weight'])

process_cfg = Preprocessing(shape=cfg['preprocessing']['shape'],
                            use_letter=cfg['preprocessing']['use_letter'],
                            normalize=cfg['preprocessing']['normalize'])

train_cfg = TrainCfg(cfg['train']['epochs'],
                     cfg['train']['learning_rate'],
                     cfg['train']['optimizer_name'],
                     cfg['train']['device'],
                     cfg['train']['resume'],
                     cfg['train']['lrsche_name']
                     )

model_cfg = ModelCfg(cfg['model']['name'],
                     cfg['model']['num_classes']
                     )

finetune_cfg = FinetuneCfg(
    strategy=cfg['finetune']['strategy'],       # 使用关键字传参更安全
    head_lr=cfg['finetune']['head_lr'],
    backbone_lr=cfg['finetune']['backbone_lr'],
    maxlen=cfg['finetune']['maxlen'],         # 对应 yaml 的 maxlen
    epoch=cfg['finetune']['epoch']            # 对应 yaml 的 epoch
)


def fun(**dic):
    print(dic)


if __name__ == '__main__':
    dic = asdict(finetune_cfg)

    fun(**dic)
