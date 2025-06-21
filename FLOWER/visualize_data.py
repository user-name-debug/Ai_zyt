import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

data_dir = 'D:/AI_project/FLOWER/flower_data/train'
class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# 预处理方法
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

orig_means, orig_stds = [], []
proc_means, proc_stds = [], []

for c in class_names:
    img_files = os.listdir(os.path.join(data_dir, c))[:5]  # 每类取5张
    for f in img_files:
        img_path = os.path.join(data_dir, c, f)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img) / 255.0
        orig_means.append(arr.mean())
        orig_stds.append(arr.std())

        # 预处理后
        img_tensor = transform(img)
        arr_proc = img_tensor.permute(1,2,0).numpy()
        proc_means.append(arr_proc.mean())
        proc_stds.append(arr_proc.std())

# 绘制对比散点图
plt.figure(figsize=(7,6))
plt.scatter(orig_means, orig_stds, alpha=0.6, color='green', label='原始图片')
plt.scatter(proc_means, proc_stds, alpha=0.6, color='blue', label='预处理后图片')
plt.xlabel('均值')
plt.ylabel('标准差')
plt.title('图片均值与标准差散点图（处理前后对比）')
plt.legend()
plt.tight_layout()
plt.show()