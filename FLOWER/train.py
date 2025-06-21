import timm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os.path as osp
import os
from pylab import mpl
from torchvision import datasets
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, recall_score
from torchutils import MetricMonitor, calculate_f1_macro, calculate_recall_macro, accuracy, \
    Uniform_picture

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示的中文字体
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
train_sum=0
test_sum=0

#输出每个分组（训练 / 测试）中分别包含多少张图像
train_path = "D:/AI_project/FLOWER/flower_data/train/"
test_path = "D:/AI_project/FLOWER/flower_data/test/"
image_list=['贝母','黄花九轮草','卷丹百合','蓝铃花','铃兰','三色堇','水仙','向日葵','雪花莲','鸢尾花']
for i in range(0,len(image_list)): # 遍历每个类别名称
    train_sum += (len(os.listdir(train_path + image_list[i])))
    test_sum += (len(os.listdir(test_path + image_list[i])))
    print("===================================================================")
    print("{}{}:{}".format(image_list[i],'训练集的数量为', train_sum))
    print("{}{}:{}".format(image_list[i],'测试集的数量为', test_sum))
    train_sum = 0
    test_sum = 0

data_path = r"D:/AI_project/FLOWER/flower_data/" #数据集路径
# 超参数（模型的配置参数）设置
params = {
    'model': 'resnet50d',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # 训练集路径
    "val_dir": osp.join(data_path, "valid"),  # 验证集路径
    'device': device,  # 设备
    'lr': 0.001,  # 设置学习率为0.001，学习率越低过程越慢，但更稳定
    'batch_size': 4,  # 单次传递给程序用以训练的数据个数
    'num_workers': 0,  # 进程数
    'epochs': 25,  # 训练轮数
    "save_dir": "../model/",  #模型保存路径
    "pretrained": True,  # 使用预训练模型
     "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 0.00001  # 学习率衰减，设置为 0.00001，用于防止模型过度拟合
}

# 在机器学习中，预训练模型是一种已经训练好的模型，用于某一任务的模型。例如，在计算机视觉任务中，预训练模型可以是从ImageNet数据集上训练出来的模型，或者是从其他数据集上训练出来的模型。
# 使用预训练模型的好处在于，它们已经在大型数据集上被训练好了，可以为我们的任务所用。我们可以使用这些预训练模型的权值作为自己模型的初始权值，然后对模型进行微调，从而达到更好的泛化能力。
# 定义神经网络模型
class SELFMODEL(nn.Module):
    # 初始化模型构造器,三个参数分别是模型的名称、输出特征数量和是否使用预训练的权值
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],pretrained=True):
        super().__init__()  # 初始化基类
        self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型(params)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features)  # 将全连接层的输出特征数量修改为类别数
    def forward(self, x):  # 卷积神经网络的前向传播
        x = self.model(x)
        return x

# 定义训练流程(训练数据的加载器,定义的模型,损失函数,优化器,训练的轮数)
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型,可以更新模型的参数
    # nBatch = len(train_loader)
    stream = tqdm(train_loader)# 设置进度条

    for i, (images, target) in enumerate(stream, start=1):  # 开始训练,遍历训练数据集中的数据和标签
        images = images.to(params['device'], non_blocking=True)  # 将数据和标签加载到设备上输入到模型中
        target = target.to(params['device'], non_blocking=True)
        output = model(images)  # 数据放入模型进行前向传播
        loss = criterion(output, target.long())  # 计算模型预测和真实标签的损失
        Probability = torch.softmax(output, dim=1)  # 计算output中每个类别的概率
        Probability = torch.argmax(Probability, dim=1).cpu()  # 找到概率最大的类别
        target = target.cpu() # 将target从GPU设备转移到CPU上
        f1_macro = f1_score(target, Probability, average='macro')  # 计算f1分数
        recall_macro = recall_score(target, Probability, average="macro", zero_division=0)  # 计算recall分数
        acc = accuracy_score(target, Probability)  # 计算准确率
        metric_monitor.update('loss', loss.item())  # 更新损失监视器
        metric_monitor.update('F1', f1_macro)  # 更新f1监视器
        metric_monitor.update('Recall', recall_macro)  # 更新召回率
        metric_monitor.update('acc', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播,调整神经网络权值参数，可以让输出更接近预期
        optimizer.step()  # 更新优化器
        # lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. Train.{metric_monitor}".format(epoch=epoch,metric_monitor=metric_monitor) #MetricMonitor存储训练过程中的各种指标
        )
    return metric_monitor.metrics['acc']["avg"], metric_monitor.metrics['loss']["avg"]  # 返回平均准确度和平均损失

# 定义模型验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 跟踪不同指标的值
    model.eval()  # 模型设置为验证格式,验证模式下模型不会更新权值
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1): # 将每次迭代的元素解压缩到变量images和target
            images = images.to(params['device'], non_blocking=True)  # 读取图片放到设备上
            target = target.to(params['device'], non_blocking=True)  # 同上,第二个参数表示传输在后台进行，可以加速传输的过程
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('acc', acc)
            # 设置进度条的显示内容
            stream.set_description("Epoch: {epoch}. val. {metric_monitor}".format(epoch=epoch,metric_monitor=metric_monitor))
    return metric_monitor.metrics['acc']["avg"], metric_monitor.metrics['loss']["avg"]

# 展示训练过程的曲线图
def show_loss_acc(acc, loss, val_acc, val_loss, sava_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    plt.figure(figsize=(8, 8))  # 创建一个新图表，并设置大小为 (8, 8)
    plt.subplot(2, 1, 1)  # 将图表分成 2 行 1 列的网格，并选择第一个子图
    plt.plot(acc, label='Training Accuracy')  # 绘制训练精度的曲线
    plt.plot(val_acc, label='Validation Accuracy')  # 绘制验证精度的曲线

    plt.legend(loc='lower right')  # 添加图例
    plt.ylabel('Accuracy')  # 设置 y 轴的标签
    plt.ylim([min(plt.ylim()), 1])  # 设置 y 轴的范围
    plt.title('训练和验证的准确性')  # 设置图表的标题

    plt.subplot(2, 1, 2)  # 将图表分成 2 行 1 列的网格，并选择第二个子图
    plt.plot(loss, label='Train_Loss')  # 绘制训练损失的曲线
    plt.plot(val_loss, label='Val_Loss')  # 绘制验证损失的曲线
    plt.legend(loc='upper right')  # 添加图例
    plt.ylabel('Cross Entropy')  # 设置 y 轴的标签
    plt.title('训练和验证的损失')
    plt.xlabel('epoch')  # 设置 x 轴的标签

    # 保存图片在savedir目录下。
    save_path = osp.join(save_dir, "train_result.png")
    # 保存图表到文件，分辨率设置为100
    plt.savefig(save_path, dpi=100)


if __name__ == '__main__':
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    data_transforms = Uniform_picture(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    save_dir = osp.join(params['save_dir'], params['model']+"_pretrained_" + str(params["img_size"]))  # 设置模型保存路径
    if not osp.isdir(save_dir):
        # 如果保存路径不存在的话就创建
        os.makedirs(save_dir)
        print("save dir {} created".format(save_dir))

    train_loader = DataLoader(
        # 按照批次加载训练集
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        # 创建一个数据加载器
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(  # 按照批次加载验证集
        valid_dataset, batch_size=params['batch_size'], shuffle=False,num_workers=params['num_workers'], pin_memory=True,
    )
    print(train_dataset.classes)

    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],pretrained=params['pretrained']) # 加载模型
    model = model.to(params['device'])  # 模型部署到设备上
    criterion = nn.CrossEntropyLoss().to(params['device'])  # 设置损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])  # 设置优化器

    # 开始训练，循环指定参数中的 epoch 次数
    for epoch in range(1, params['epochs'] + 1):
        # 训练模型一个 epoch 并获取准确率和损失
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        # 将准确率和损失值添加到相应的列表中
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        save_path = osp.join(save_dir, f"acc{acc:.5f}_weights.h5")
        torch.save(model.state_dict(), save_path)# 保存文件
        best_acc = val_acc #将当前模型的权重保存到save_dir目录下的文件中
    # 显示训练和验证集的损失和准确率折线图
    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))