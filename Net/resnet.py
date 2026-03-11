import torch as th
from torch import nn
from torch.nn import functional as f
class Residual(nn.Module):
    def __init__(self, ins, ous, stride=1, shortcut=None):
        super().__init__()

        self.left = nn.Sequential(
            nn.Conv2d(ins, ous, 3, stride, 1, bias=False),
            nn.BatchNorm2d(ous),
            nn.ReLU(),
            nn.Conv2d(ous, ous, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ous))

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return f.relu(out)


class ResNet(nn.Module):
    def __init__(self,num_classes,layer_nums=[3,4,6,4]):
        super().__init__()

        init_chanle = 64
        self.pre = nn.Sequential(
            nn.Conv2d(3, init_chanle, 7, 2, 3, bias=False),
            nn.BatchNorm2d(init_chanle),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))

        layers = []
        for i, num in enumerate(layer_nums):
            if i == 0:
                layers.append(self._make_layer(
                    init_chanle, init_chanle, 1, num))
            else:
                layers.append(self._make_layer(
                    init_chanle, init_chanle*2, 2, num))
                init_chanle *= 2

        self.layers = nn.Sequential(*layers)
        self.drop_out=nn.Dropout(0.3)
        self.fc = nn.Linear(init_chanle, num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self, ins, ous, stride, num):

        if stride != 1 or ins != ous: 
            shortcut = nn.Sequential(
                nn.Conv2d(ins, ous, 1, stride, 0, bias=False),
                nn.BatchNorm2d(ous))
        else:
            shortcut = None

        layers = []
        layers.append(Residual(ins, ous, stride, shortcut))
        for i in range(1, num):
            layers.append(Residual(ous, ous))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layers(x)
        x = f.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x=self.drop_out(x)

        return self.fc(x)

#blocks [2,2,2,2] resnet18
#blocks [3,4,6,3] resnet34
if __name__ == '__main__':
    
    net=ResNet(10)
    





