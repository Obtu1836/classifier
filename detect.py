from pathlib import Path
import inspect
import Net
import torch as th
from torch import nn 
import cv2
from config.Config import path_cfg,model_cfg,get_device
from utils.process import val_forms
from torch.nn import functional as f


class Detect:
    def __init__(self,device='cpu'):
        
        self.device=get_device(device)


    def get_net(self,name: str):

        classes = inspect.getmembers(Net, inspect.isclass)
        funs = inspect.getmembers(Net, inspect.isfunction)
        call_able = dict(classes+funs)
        all_net = {name: model for (name, model) in call_able.items()}
        num_class = model_cfg.num_classes
        net: nn.Module = all_net[name](num_class)
        check_path = path_cfg.checkpoint/f"{name}_best.pt"
        check = th.load(check_path, map_location=self.device)
        weight = check['model']
        net.load_state_dict(weight)
        class_information = check['class_information']
        
        self.net=net
        self.information=class_information

    @th.no_grad()
    def run(self,img_path):

        img=cv2.imread(img_path)
        if img is not None:
            img_tensor=val_forms(img[:,:,::-1])
            img_tensor=img_tensor.to(self.device) #type: ignore
            output=self.net(img_tensor[None,...])
            proba=f.softmax(output,dim=1)
            idx=th.argmax(proba,dim=1).item()
            text=self.information[idx]
            h,w=img.shape[:2]
            cv2.putText(img,text,(w//4,h//2),fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.8,color=(0,255,0),thickness=2)
            cv2.imshow('img',img)
            cv2.waitKey(0)


def main():

    detect=Detect('cpu')
    detect.get_net("PretrainedDensenet121")

    detect.run(r'imgs\train\chicken\17.jpeg')

if __name__ == '__main__':
    main()
