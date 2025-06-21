import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import SELFMODEL
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
model_path = r"D:/AI_project/model/resnet50d_pretrained_224/acc0.99464_weights.h5"  # 模型权重路径
test_dir = r"D:/AI_project/FLOWER/flower_data/test"
img_size = 224

# 图像预处理
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
test_dataset = datasets.ImageFolder(test_dir, test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 获取类别名和类别数
class_names = test_dataset.classes
num_classes = len(class_names)

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model = SELFMODEL(model_name='resnet50d', out_features=num_classes, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 推理，收集真实标签和softmax概率
all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

# 多分类标签二值化
y_true_bin = label_binarize(all_labels, classes=range(num_classes))
y_score = np.array(all_probs)

# 计算并绘制ROC曲线
plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('多分类ROC曲线')
plt.legend()
plt.tight_layout()
plt.show()