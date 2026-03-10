import torch as th
from torch import nn 

from torchvision.models import densenet121
from torchinfo import summary

'''
继承sequeential 可以不用实现forward函数 因为内部已经实现了
moduledict modulelist module 都需要自己实现forward函数

DenseNet-121：grow=32, bn=4 [6,12,24,16]
DenseNet-169：grow=32, bn=4 [6,12,32,32]
DenseNet-201：grow=32, bn=4 [6,12,48,32]
DenseNet-161：grow=48, bn=4 [6,12,36,24]

'''

class DenseLayer(nn.Sequential):
    '''
    layer层 总体是降维的 在初期第一个卷积可能是升维 到后面的话 就是降维的作用
    所有layer层的最终输出通道 都是grow 常用数为 32
    '''
    def __init__(self,ins,bn,grow,drop):
        super().__init__()
        '''
        同一个容器使用add_module 名字要区分 不然会覆盖
        '''
        self.add_module('norm',nn.BatchNorm2d(ins))
        self.add_module('relu',nn.ReLU())
        self.add_module('conv',nn.Conv2d(ins,bn*grow,1,1,0,bias=False))

        self.add_module('norm1',nn.BatchNorm2d(bn*grow))
        self.add_module('relu1',nn.ReLU())
        self.add_module('conv1',nn.Conv2d(bn*grow,grow,3,1,1,bias=False))

        if drop>0:
            self.add_module('drop',nn.Dropout(drop))
    
class DenseBlocks(nn.ModuleDict):
    '''
    block 每个block 包含n个layer 这个层是密集连接的关键
    通过cat 循环拼接每个layer的输出 达到特征重用的目的 
    '''
    def __init__(self,nums,ins,bn=4,grow=32,drop=0):
        super().__init__()

        for i in range(nums):#每个block包含nums个layer
            layer=DenseLayer(ins+i*grow,bn,grow,drop)
            self.add_module(f'layer{i+1}',layer)

    def forward(self,x):

        for _,layer in self.items(): # 循环拼接 每个layer的输出
            out=layer(x)
            x=th.cat([x,out],dim=1)
        
        return x
    
class DenseTransition(nn.Sequential):
    # 通过1v1卷积降低维度为原来的一半 同时通过avgpool 降低特征图的一半
    # 每个block之间 加入这个层 
    def __init__(self,ins):
        super().__init__()

        self.add_module('norm',nn.BatchNorm2d(ins))
        self.add_module('relu',nn.ReLU())
        self.add_module('conv',nn.Conv2d(ins,ins//2,1,1,0,bias=False))
        self.add_module('avgpool',nn.AvgPool2d(2,2))

class DenseNet(nn.Module):
    def __init__(self,num_blocks,grow=32,bn=4,drop=0,num_class=1000):
        super().__init__()

        init_dims=64 #layer1 和resnet一样
        self.layer1=nn.Sequential()
        self.layer1.add_module('conv',nn.Conv2d(3,init_dims,7,2,3,bias=False))
        self.layer1.add_module('norm',nn.BatchNorm2d(init_dims))    
        self.layer1.add_module('relu',nn.ReLU())
        self.layer1.add_module('maxpool',nn.MaxPool2d(3,2,1))


        self.layer2=nn.Sequential()
        for i,num in enumerate(num_blocks):#循环生成block模块
            block=DenseBlocks(num,init_dims,bn,grow,drop)
            self.layer2.add_module(f'DenseBlock{i+1}',block)
            # 每个block 包含n个layer 在block会拼接n个layer的输出
            # 每个layers输出grow个特征图 所以每个block最终输出通道最终会是输入通道+num*grow
            init_dims=init_dims+num*grow 

            if i!=len(num_blocks)-1:
                trans=DenseTransition(init_dims)#降维
                self.layer2.add_module(f'DenseTrans{i+1}',trans)
                init_dims//=2# 保持降维后的通道
        
        self.layer3=nn.Sequential()
        self.layer3.add_module('norm',nn.BatchNorm2d(init_dims))
        self.layer3.add_module('relu',nn.ReLU())
        self.layer3.add_module('avgpool',nn.AdaptiveAvgPool2d((1,1)))
        self.layer3.add_module('flatten',nn.Flatten(1))

        self.fc=nn.Linear(init_dims,num_class)
    
    def forward(self,x):

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        return self.fc(x)
    
if __name__ == '__main__':

    net=DenseNet([6,12,24,16],32)
    net2=densenet121()
    summary(net,(1,3,224,224))
    



            




        

