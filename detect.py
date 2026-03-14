import cv2
import torch as th
from torch.nn import functional
from train import Factory
from utils.process import val_forms
# from utils import cfg
from config.Config import model_cfg, path_cfg
from pathlib import Path

device = 'mps' if th.backends.mps.is_available() else 'cuda'


# cfg_model=cfg['model'
net = Factory.create_model(model_cfg)
path = Path(path_cfg.checkpoint)/f"{model_cfg.name}_best.pt"
check = th.load(path, map_location='cpu')
net.load_state_dict(check['model'])
net.eval()
class_names = check['class_information']


def detect_one_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise IOError('无法读取照片')

    img2 = val_forms(img[:, :, ::-1])

    with th.no_grad():
        output = net(img2[None, ...])
    proba = functional.softmax(output, dim=1)

    pred_class = th.argmax(proba, dim=1)
    print(pred_class)
    text = class_names[pred_class.item()]

    # BGR 格式：(0, 0, 255) 是红色，(0, 255, 0) 是绿色
    ims = cv2.putText(img, text, (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('img', ims)
    cv2.waitKey(0)


detect_one_img(r'imgs/train/chicken/6.jpeg')
