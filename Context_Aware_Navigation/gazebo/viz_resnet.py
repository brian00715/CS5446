import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True).to(device)
model.eval()  # 设置为评估模式

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载输入图像
from PIL import Image

img = Image.open("/home/simon/Pictures/image.png")
inp = transform(img).unsqueeze(0).to(device)

# 前向传播获取中间层输出
outputs = []
names = []


def hook(module, input, output):
    outputs.append(output)
    names.append(module.__class__.__name__)


# 注册hook函数监听模型各层输出
for name, module in model.named_modules():
    if "conv" in name:
        module.register_forward_hook(hook)

# 前向传播
_ = model(inp)

# 可视化第一个卷积层的特征图
plt.figure(figsize=(20, 17))
for i, feat in enumerate(outputs[0][0]):  # 第一个输出元素对应第一个卷积层
    if i == 40:
        break
    plt.subplot(5, 8, i + 1 )
    plt.imshow(feat.detach().numpy(), cmap="gray")
    plt.axis("off")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
