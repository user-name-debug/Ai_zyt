import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import SELFMODEL
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
model_path = r"D:/AI_project/model/resnet50d_pretrained_224/acc0.99464_weights.h5"  # 模型权重路径
test_dir = r"D:/AI_project/FLOWER/flower_data/test"
num_classes = 10
img_size = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
test_dataset = datasets.ImageFolder(test_dir, test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# 获取类别名（顺序与ImageFolder一致，防止标签错乱）
class_names = test_dataset.classes

# 加载模型
model = SELFMODEL(model_name='resnet50d', out_features=num_classes, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 随机取一批图片进行可视化
imgs, labels = next(iter(test_loader))
imgs = imgs.to(device)
with torch.no_grad():
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# 反归一化用于显示
def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1,2,0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# 把所有图片拼成一行，真实和预测标签分别作为标题
plt.figure(figsize=(20, 5))
for i in range(len(imgs)):
    plt.subplot(2, len(imgs), i+1)
    plt.imshow(denormalize(imgs[i]))
    plt.title(f"真实: {class_names[labels[i]]}", fontsize=10)
    plt.axis('off')
    plt.subplot(2, len(imgs), len(imgs)+i+1)
    plt.imshow(denormalize(imgs[i]))
    plt.title(f"预测: {class_names[preds[i]]}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()