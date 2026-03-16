from torchvision.models import resnet18, ResNet18_Weights, densenet121, DenseNet121_Weights
from torch import nn
import torch as th

'''
添加自定义的网络时 仿照已有的模板写 函数名称必须以大写的Pretrained+名称,(名称首字母大小写无所谓) 但是尽量保持统一 

比如新导入 vgg 的预训练模型时 

def PretrainedVgg...

然后在Net/__init__.py中 导入
最后在config.yaml里 model.name 写上 将函数名称
'''


def PretrainedResnet18(num_classes: int) -> nn.Module:

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    ins = model.fc.in_features
    model.fc = nn.Linear(ins, num_classes)

    return model


def PretrainedDensenet121(num_classes: int) -> nn.Module:

    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    ins = model.classifier.in_features

    model.classifier = nn.Linear(ins, num_classes)

    return model


if __name__ == '__main__':

    data = th.rand(1, 3, 224, 224)
    model = PretrainedResnet18(10)

    print(model.__class__.__name__)
