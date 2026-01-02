## 코드 전문
```python
# 실행마다 동일한 결과를 얻기 위해 파이토치에 랜덤 시드를 지정하고 GPU 연산을 결정적으로 만든다.
import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
     
# 2절에서 만든 모델 다운로드
!wget https://github.com/rickiepark/hg-mldl2/raw/refs/heads/main/best_cnn_model.pt

import torch.nn as nn

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))
     
model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))

# childeren(): 모델에 추가된 층을 반환하는 파이썬 제너레이터(generator) 객체를 반환
layers = [layer for layer in model.children()]
     

print(layers[0])
     
# Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)

#Sequential() 클래스로 만든 모델은 정수 인덱스를 통해 층을 참조할 수 있다.
model[0]
     
# Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)

for name, layer in model.named_children():
    print(f"{name:10s}", layer) # 문자열 name을 최소 10칸의 폭으로 출력하라
'''     
conv1      Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
relu1      ReLU()
pool1      MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
conv2      Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
relu2      ReLU()
pool2      MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
flatten    Flatten(start_dim=1, end_dim=-1)
dense1     Linear(in_features=3136, out_features=100, bias=True)
relu3      ReLU()
dropout    Dropout(p=0.3, inplace=False)
dense2     Linear(in_features=100, out_features=10, bias=True)
'''

model.conv1
     
# Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)

conv_weights = model.conv1.weight.data

# PyTorch의 tensor는 numpy와 유사한 기능을 많이 제공한다.
print(conv_weights.mean(), conv_weights.std())
     
# tensor(-0.0550) tensor(0.3589)

import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

print(conv_weights.shape)
     
# torch.Size([32, 1, 3, 3])

fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[i*16 + j,0,:,:], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

from torchvision.datasets import FashionMNIST

fm_train = FashionMNIST(root='.', train=True, download=True)
train_input = fm_train.data

plt.imshow(train_input[0], cmap='gray_r')
plt.show()


ankle_boot = train_input[0:1].reshape(1, 1, 28, 28) / 255.0

model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
     

print(feature_maps.shape)
     
# torch.Size([1, 32, 28, 28])

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()

model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
    feature_maps = model.pool1(feature_maps)
    feature_maps = model.conv2(feature_maps)
    feature_maps = model.relu2(feature_maps)
     

model.eval()
x = ankle_boot
with torch.no_grad():
    for name, layer in model.named_children(): # named_chidren(): (이름, 서브모듈)을 반환함
        x = layer(x)
        if name == 'relu2':
            break
feature_maps = x
     

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()
```
