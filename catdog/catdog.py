import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.nn.functional as F
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小为256x256
    transforms.ToTensor(),  # 将图片转换为Tensor
])
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
device = torch.device("cuda:0")
model = SimpleCNN()
model.load_state_dict(torch.load('E:\量化\MyKaggle\catdog\cnn.pth'))

# 将模型设置为评估模式
model.eval()

# 确保模型在正确的设备上
model.to(device)
import os
from PIL import Image

test_path = 'E:/Kaggle/dog vs cat/test1/test1'
image_lists = os.listdir(test_path)
pre = []
for image_list in image_lists:
    index = os.path.splitext(image_list)[0]
    image_path = os.path.join(test_path, image_list)
    image = Image.open(image_path)
    image = image.convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    pre.append([index, predicted.item()])