import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
from train import SELFMODEL
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示的中文字体
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)  # 设置 NumPy 的随机数生成器的种子
torch.manual_seed(seed)  # 设置 PyTorch CPU 版本的随机数生成器的种子
torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 版本的随机数生成器的种子
torch.backends.cudnn.deterministic = True  # 每次返回的卷积算法将是默认算法
torch.backends.cudnn.benchmark = True  # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
# 数据集根目录
data_path = r"D:/AI_project/FLOWER/flower_data/"
model_path = r"D:/AI_project/model/resnet50d_pretrained_224/acc0.99464_weights.h5"  # 模型权重路径
model_name = 'resnet50d'
# 数据集训练时输入模型的大小
img_size = 224

# 超参数（模型的配置参数）设置
params = {
    'model': model_name,  # 选择预训练模型
    "img_size": img_size,  # 图片输入的大小
    "test_dir": osp.join(data_path, "test"),  # 测试集子目录
    'batch_size': 4,  # 单次传递给程序用以训练的数据个数
    'num_workers': 0,  # 进程数
    "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
}

#对输入的图像数据集进行验证
def test(val_loader, model, params, class_names):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    with torch.no_grad():  # 禁用自动求导，加速推理过程
        for i, (images, target) in enumerate(stream, start=1): # 验证数据集中的所有图像和标签
            output = model(images)  # 向前传播
            target_numpy = target.cpu().numpy()  # 将标签从 GPU 复制到 CPU 上并转化为 numpy 数组
            y_pred = torch.softmax(output, dim=1)# 使用 softmax 函数将模型的输出转化为概率
            # 使用 argmax 函数获取概率最大的标签，并将其从 GPU 复制到 CPU 上并转化为 numpy 数组
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            # 将真实标签和预测标签添加到列表中
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc

            #f1_macro多分类问题，不受数据不平衡影响，容易受到识别性高（高recall、高precision）的类别影响
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            # 设置进度条的描述文本
            stream.set_description(
                "mode: {epoch}.{metric_monitor}".format(epoch="test",metric_monitor=metric_monitor)
            )
    # 返回准确率、F1 值、召回率的平均值
    return metric_monitor.metrics['Accuracy']["avg"], \
           metric_monitor.metrics['F1']["avg"], \
           metric_monitor.metrics['Recall']["avg"]

if __name__ == '__main__':
    data_transforms = Uniform_picture(img_size=params["img_size"])  # 获取图像预处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    test_dataset = datasets.ImageFolder(params["test_dir"], valid_transforms)  # 加载测试数据集
    class_names = test_dataset.classes  # 获取类别名称
    print(class_names)

    # 按照批次将测试数据集加载到内存中
    test_loader = DataLoader(
        test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=True,
    )
    # 加载模型，实例化自定义模型类，并将预训练的模型的参数设置为False
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],pretrained=False)
    weights = torch.load(model_path) # 加载模型的权重
    model.load_state_dict(weights) # 将权重应用到模型上
    model.eval()  # 将模型设置为测试模式
    acc, f1, recall = test(test_loader, model, params, class_names)  # 指标上的测试结果,返回准确率、F1值和召回率
    print("测试结果：")
    print(f"acc: {acc},recall: {recall}")