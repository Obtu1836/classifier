import torch as th
from torch import nn
from torch.nn import functional as f

'''
权重的形状为类别的形状 权重元素各个样本占样本总数的比例的倒数

作用是  样本不均衡时 防止模型偏向于样本多的类别
'''

def torchAchieve(probas,target,weight):
    criterion=nn.CrossEntropyLoss(weight=weight)
    loss=criterion(probas,target)
    
    return loss.item()

def selfAchive(probas,target,weight):
    probas=f.log_softmax(probas,dim=1)
    wght=weight[target]
    k_class=len(th.unique(target))
    mask=f.one_hot(target,num_classes=k_class)

    res=-((mask*probas)*wght[:,None]).sum()/wght.sum()
    return res.item()

if __name__ == '__main__':
    k=3
    probas=th.rand(5,k)
    target=th.tensor([0,1,2,1,0])
    weight=th.rand(k)
    res1=torchAchieve(probas,target,weight)
    print(res1)
    res2=selfAchive(probas,target,weight)
    print(res2)



