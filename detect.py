import cv2
import Net
import inspect
import torch as th
from pathlib import Path
from numpy.typing import NDArray
from torch import nn
from typing import cast
from torch.nn import functional as f
from config.Config import path_cfg, model_cfg, get_device
from utils.process import val_forms
from loguru import logger

class Detect:
    def __init__(self, device='cpu'):

        self.device = get_device(device)
        logger.success(f"{self.device}")

    def get_net(self, model_name: str):

        classes = inspect.getmembers(Net, inspect.isclass)
        funs = inspect.getmembers(Net, inspect.isfunction)
        call_able = dict(classes+funs)
        all_net = {model_name: model for (
            model_name, model) in call_able.items()}
        num_class = model_cfg.num_classes
        net: nn.Module = all_net[model_name](num_class)
        check_path = path_cfg.checkpoint/f"{model_name}_best.pt"
        check = th.load(check_path, map_location=self.device)
        weight = check['model']
        net.load_state_dict(weight)
        class_information = check['class_information']

        self.net = net
        self.net.eval()
        self.net.to(self.device)
        self.information = class_information

    @th.no_grad()
    def detect_single_img(self, img_path: Path,time=0):

        img = self._readimg(img_path)
        img_tensor = cast(th.Tensor, val_forms(img[:, :, ::-1]))
        img_tensor = img_tensor.to(self.device)
        output = self.net(img_tensor[None, ...])
        proba = f.softmax(output, dim=1)
        idx = th.argmax(proba, dim=1).item()
        text = self.information[idx]
        h, w = img.shape[:2]
        cv2.putText(img, text, (w//4, h//2), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.8, color=(0, 255, 0), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(time)
        return text

    def _readimg(self, img_path: Path) -> NDArray:
        img = cv2.imread(img_path)
        if img is None:
            logger.error('照片读取错误')
            raise FileNotFoundError()
        return img


def main():

    detect = Detect('cuda')
    detect.get_net("PretrainedResnet18")
    
    true,num=0,0
    img_path = Path(r'imgs\val\butterfly')
    for child in img_path.iterdir():
        text=detect.detect_single_img(child,10)
        if text==img_path.name:
            true+=1
        num+=1
    print(true/num)


if __name__ == '__main__':
    main()
