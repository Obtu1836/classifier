import torch as th
from torch import nn
from torch.nn import functional as f

'''
指标平滑的作用 起到平滑的作用 
在训练初期 会相对的降低loss  在一定程度上可以减小震荡
在训练后期 会相对增大loss 抑制过渡自信 改善概率校准 使训练稳定 
在图像上 就像是在y轴方向 互相压平  使图像相对平缓

副作用: 略微降低模型在训练集的极限准确率
'''

def torchAchieve(probas,target,smoothing):

    criterion=nn.CrossEntropyLoss(label_smoothing=smoothing)
    loss=criterion(probas,target)
    return loss.item()

def selfAchieve(probas,target,smoothing):

    probas=f.log_softmax(probas,dim=1)
    k_class=len(th.unique(target))
    mask=th.eye(k_class)[target]
    mask=(1-smoothing)*mask+smoothing/k_class
    res=-(mask*probas).sum()/len(probas)

    return res.item()



if __name__ == '__main__':
    k=3
    probas=th.rand(5,k)
    target=th.tensor([0,1,2,1,0])
    smoothing=0.1

    res1=torchAchieve(probas,target,smoothing)
    res2=selfAchieve(probas,target,smoothing)
    print(res1)
    print(res2)