import torch
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torchvision import datasets, models, transforms

# 数据增强,统一图片的格式。
def Uniform_picture(img_size=224):
    # 创建用于训练和验证的两组图像转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 对图像尺寸进行调整，让模型输入大小统一为224*224
            transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转概率为0.2
            transforms.RandomRotation((-5, 5)),  # 随机旋转，旋转角度在 -5 到 5 度之间
            transforms.RandomAutocontrast(p=0.2),  # 随机自适应对比度增强概率为0.2
            transforms.ToTensor(),  # 图像转换为tensor格式,方便模型对的处理和计算,可以方便的在GPU上进行计算
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像标准化,使得模型的训练更加稳定,3个值对应RGP三个通道
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# 计算acc
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)  #输出中每个类别的最大值
    y_pred = torch.argmax(y_pred, dim=1).cpu()  #找到最大值的位置
    target = target.cpu()
    return accuracy_score(target, y_pred)  #计算分类准确率


# 计算F1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)  #计算output中每个类别的概率
    y_pred = torch.argmax(y_pred, dim=1).cpu()  #找到概率最大的类别
    target = target.cpu()
    return f1_score(target, y_pred, average='macro') #计算平均F1分数


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    return recall_score(target, y_pred,average="macro", zero_division=0)  #计算宏召回率，对每个类别分别计算召回率

# 训练的时候输出信息使用，计算训练过程中的指标平均值
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset() # 重置计数器

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name] #指定指标的名称
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):  #格式化为字符串
        return " | ".join(
            ["{metric_name}: {avg:.{float_precision}f}".format(metric_name=metric_name, avg=metric["avg"],float_precision=self.float_precision)
                for (metric_name, metric) in self.metrics.items()
            ]
        )