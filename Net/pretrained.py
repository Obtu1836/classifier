from torchvision.models import ResNet18_Weights, resnet18, densenet121, DenseNet121_Weights
from torch import nn
import torch as th


def PretrainedResnet18(num_classes):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    ins = model.fc.in_features
    model.fc = nn.Linear(ins, num_classes)

    return model


def PretrainedDensenet121(num_classes):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    ins = model.classifier.in_features

    model.classifier = nn.Linear(ins, num_classes)

    return model


if __name__ == '__main__':

    data = th.rand(1, 3, 224, 224)
    model = PretrainedResnet18(10)

    print(model(data).shape)
