import torch as th
from torch.nn import functional as f
from detect import Detect

from utils.dataload import AnimalDataLoader
from config.Config import path_cfg, get_device, dataset_cfg, model_cfg

from torchmetrics import (ConfusionMatrix, Accuracy,
                          Recall, Precision, F1Score, Metric,
                          PrecisionRecallCurve, AveragePrecision,
                          ROC, AUROC)
from loguru import logger
from torchmetrics.classification import MultilabelConfusionMatrix

import matplotlib.pyplot as plt


class CalculateMeric(Detect):
    def __init__(self, device, num_class: int):
        super().__init__(device)

        self.acc = Accuracy(task='multiclass',
                            num_classes=num_class, average='macro').to(device)
        self.precision = Precision(
            'multiclass', num_classes=num_class, average='macro').to(device)
        self.recall = Recall(
            'multiclass', num_classes=num_class, average='macro').to(device)
        self.confusion = ConfusionMatrix(
            'multiclass', num_classes=num_class).to(device)
        self.f1 = F1Score('multiclass', num_classes=num_class,
                          average='macro').to(device)

        self.single_class_score = MultilabelConfusionMatrix(
            num_labels=num_class).to(device)

        self.pr_probas = PrecisionRecallCurve(
            task='multiclass', num_classes=num_class).to(device)
        self.ap_probas = AveragePrecision(
            'multiclass', num_classes=num_class, average='none').to(device)
        self.roc_probas = ROC(
            task='multiclass', num_classes=num_class).to(device)
        self.auc_probas = AUROC(
            'multiclass', num_classes=num_class, average='none').to(device)

        self._init_attrs = [(k, v) for k, v in vars(
            self).items() if isinstance(v, Metric)]

        self.num_class = num_class

    def load_val_loader(self):

        val_loader, _, _ = AnimalDataLoader(
            'val', path_cfg, dataset_cfg).get_loader()
        return val_loader

    @th.no_grad()
    def _evaluate(self, val_load):
        for tensor, label in val_load:
            tensor, label = tensor.to(self.device), label.to(self.device)
            output = self.net(tensor)
            output = f.softmax(output, dim=1)
            pred = th.argmax(output, dim=1)

            for name, cls in self._init_attrs:
                if name == 'single_class_score':
                    l_pred = f.one_hot(pred, self.num_class)
                    l_label = f.one_hot(label, self.num_class)
                    cls.update(l_pred, l_label)
                elif 'probas' in name:
                    cls.update(output, label)
                else:
                    cls.update(pred, label)

    @property
    def acc_score(self):
        return round(self.acc.compute().item(), 3)

    @property
    def recall_score(self):
        return round(self.recall.compute().item(), 3)

    @property
    def precision_score(self):
        return round(self.precision.compute().item(), 3)

    @property
    def confusion_matrix(self):
        return self.confusion.compute()

    @property
    def f1_score(self):
        return round(self.f1.compute().item(), 3)

    def cal_perclass_metirc(self):
        per_class = self.single_class_score.compute()
        for i, tensor in enumerate(per_class):
            logger.info(f'第{i+1}个类别 {self.information[i]}:')
            rec = (tensor[1, :]/tensor[1, :].sum())[1]
            prec = (tensor[:, 1]/tensor[:, 1].sum())[1]
            logger.info(f'召回率:{round(rec.item(), 3)}')
            logger.info(f'精确率:{round(prec.item(), 3)}')

    def cal_pr_roc(self):
        precision, recall, _ = self.pr_probas.compute()
        aps = self.ap_probas.compute()
        fpr, tpr, _ = self.roc_probas.compute()
        aucs = self.auc_probas.compute()

        paint = Paint(self.information)
        pr_data = [recall, precision, aps]
        roc_data = [fpr, tpr, aucs]

        paint.draw(pr_data, roc_data)


class Paint:
    def __init__(self, class_information):

        self.class_information = class_information
        self.fig, self.ax = plt.subplots(1, 2, sharey=True, figsize=(12, 4),)

    def draw(self, pr: list[th.Tensor], roc: list[th.Tensor]):

        length = len(pr[2])

        for i in range(length):
            information = self.class_information[i]
            pr_label = information+" "+str(pr[2][i].cpu().numpy().round(3))
            self.ax[0].plot(pr[0][i].cpu().numpy(), pr[1]
                            [i].cpu().numpy(), label=pr_label)
            roc_label = information+" "+str(roc[2][i].cpu().numpy().round(3))
            self.ax[1].plot(roc[0][i].cpu().numpy(), roc[1]
                            [i].cpu().numpy(), label=roc_label)

        self.ax[0].set_xlabel('recall')
        self.ax[0].set_ylabel('precision')
        self.ax[0].legend(loc='best')

        self.ax[1].set_xlabel('fpr')
        self.ax[1].set_ylabel('tpr')
        self.ax[1].legend(loc='best')

        plt.show()


def main():

    device = get_device('cuda')
    net_name = 'PretrainedDensenet121'
    met = CalculateMeric(device, model_cfg.num_classes)
    met.get_net(net_name)
    val_load = met.load_val_loader()

    met._evaluate(val_load)
    acc = met.acc_score
    recall = met.recall_score
    precision = met.precision_score
    f1 = met.f1_score
    logger.info(f"模型准确率：{acc},召回率: {recall},\
                  精准度: {precision},f1_score: {f1}")

    met.cal_perclass_metirc()
    met.cal_pr_roc()


if __name__ == '__main__':
    main()
