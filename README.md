# 🐾 Animal Classifier - 通用的图像分类开发框架

这是一个基于 PyTorch 构建的、高度可配置且易于扩展的图像分类项目。它不仅内置了经典的模型（如 ResNet, DenseNet），还支持**一键切换预训练模型**、**动态微调策略**以及**自动化的数据增强预处理**。

无论你是想快速训练一个动物分类器，还是想基于此框架开发自己的计算机视觉分类任务，这个项目都能提供极大的便利。

---

## ✨ 核心特性

- **🚀 零代码配置**：通过 `config.yaml` 即可修改训练参数、模型选择、数据路径及设备选择（支持 CUDA/MPS/CPU）。
- **🛠️ 模块化设计**：模型 (Net)、数据加载 (utils/dataload)、训练逻辑 (Trainer) 完全解耦，极易进行二次开发。
- **📈 智能训练策略**：
  - 支持 **标签平滑 (Label Smoothing)** 提高模型泛化能力。
  - 支持 **类别权重均衡**：通过 `use_sampling_weight`（采样权重）或 `use_loss_weight`（损失权重）轻松应对数据集不平衡问题。
  - 支持 **动态全参数微调 (Finetune)**：可选择不同的优化器 不同的学习率调度器 分别调参 
- **🔍 快速推理**：内置 `detect.py`，支持单张图像的快速预测与可视化展示。

- **📊 指标评价（Evaluation / Metrics）**：内置 `metric/calculate.py`，使用 `torchmetrics` 计算并输出 **准确率(Accuracy)、精确率(Precision)、召回率(Recall)、F1-score**，同时支持混淆矩阵与按类别的精/召回统计，方便诊断模型在每个类别上的表现。

---

## 📂 项目结构

```text
├── config.yaml          # 核心配置文件（所有训练参数都在这里！）
├── train.py             # 训练启动脚本
├── detect.py            # 推理/检测脚本
├── checkpoint/          # 模型权重存放目录
├── config/              # 配置解析逻辑
├── Net/                 # 网络模型定义（ResNet, DenseNet, 预训练模型等）
├── utils/               # 工具类（数据加载、预处理、转换等）
└── imgs/                # 数据集存放目录 (train/val 分离)
```

---

## 🛠️ 自定义功能指南

本项目的设计初衷就是**高可扩展性**，以下是实现自定义功能的方法：

### 1. 如何添加自己的网络模型？
无需修改复杂的训练逻辑，只需两步：

- 如果是添加预训练模型 在 `Net/pretrained.py` 中导入预训练模型函数，仿照样本结构就可以 确保函数名以 `Pretrained` 开头（例如 `PretrainedVgg`）。 如果是自己写模型 则没有上述限制 确保函数名不以 'Pretrained'开头即可

- 在 `Net/__init__.py` 中导出该函数。
- 在 `config.yaml` 的 `model.name` 中填入你的函数名即可。

### 2. 应对数据集类别不均衡
如果某些类别的图片很少，你可以在 `config.yaml` 中开启平衡策略：
- `use_sampling_weight: True`：通过加权采样，让模型在训练时更多地看到少数类别的样本。
- `use_loss_weight: True`：在计算损失时给少数类别更高的权重。

### 3. 自定义微调 (Fine-tuning) 策略
项目支持在训练中期自动“解冻”网络参数：
- 设置 `strategy: epoch` 并在 `epoch: 10` 指定在第 10 轮开启全量训练。
- 设置 `strategy: acc` 结合 `maxlen: 15`，当模型准确率在连续 15 轮内没有显著提升时，系统会自动解冻 Backbone，寻找更优解。

---

## 🚀 快速开始

### 1. 安装环境
确保你已安装 Python 3.12+
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将图片按照类别存放在 `imgs/train` 和 `imgs/val` 目录下：
```text
imgs
    train/
         cat/
         dog/
    val/
        cat/
        dog/
```

### 3. 开始训练
调整好 `config.yaml` 后，直接运行：
```bash
python train.py
```

### 4. 推理检测
```bash
python detect.py
```

### 5. 指标分析
```bash
python -m metric.calculate

---

## 📝 开发者建议
如果是从零开始一个新任务，建议先在 `config.yaml` 中选择 `PretrainedResnet18`，并将 `resume: False`。等模型在 训练稳定后，利用项目的 `finetune` 策略开启全参数微调，通常能获得更好的效果。

项目断点 继续训练 观察 网络名称 优化器名称 等是否与之前一致 否则容易出现意外的情况 其次 对于自己写的网络模型 是否使用微调 以及是否是断点训练 一定要观察清楚 防止出现不匹配的情况 尽管项目中已经做了一些处理 但是依然可能存在与期望不一致的情况



