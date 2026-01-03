## 코드 전문
```python
# 실행마다 동일한 결과를 얻기 위해 파이토치에 랜덤 시드를 지정하고 GPU 연산을 결정적으로 만든다.
import torch

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
     

from keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

from keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
     

print(train_seq.shape, train_target.shape)
# (20000, 100) (20000,)

train_seq = torch.tensor(train_seq)
val_seq = torch.tensor(val_seq)

print(train_target.dtype)
# int64

train_target = torch.tensor(train_target, dtype=torch.float32)
val_target = torch.tensor(val_target, dtype=torch.float32)
     

print(train_target.dtype)
     
torch.float32

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_seq, train_target)
val_dataset = TensorDataset(val_seq, val_target)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
     

import torch.nn as nn

'''
파이토치의 RNN 층은 마지막 순환층의 모든 타임스텝에 대한 hidden state와(마지막 RNN layer),
모든 순환층의 최종 hidden state 두 가지 값을 반환한다(layer 별로 하나씩).
따라서 Sequential 클래스를 이용해 모델을 구현하기에는 어려움이 있다.
대신 nn.Module의 서브클래스를 만들어 모델을 구현하면 손쉽게 RNN 모델을 만들 수 있다.
'''
class IMDBRnn(nn.Module):
    # 생성자
    def __init__(self):
        super().__init__() # 부모 클래스의 생성자 호출
        self.embedding = nn.Embedding(500, 16) # (어휘 사전 크기, 임베딩 벡터 크기)
        self.rnn = nn.RNN(16, 8, batch_first=True) # (임베딩 벡터 크기, 뉴런 개수)
        '''파이썬에서 RNN 클래스는 입력 차원의 순서가 (시퀀스 길이, 배치 크기, 임베딩 크기)라고 가정한다.
           그러나 이 클래스에서 임베딩 층을 통과한 출력은 배치 크기가 맨 앞에 높여 (배치 크기, 시퀀스 길이, 임베딩 크기)의
           값을 가진다. 따라서 배치 차원이 맨 앞이라는 걸 알리기 위해 batch_first = True로 지정한다.
        '''
        self.dense = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    # 정방향 계산
    def forward(self, x): 
        x = self.embedding(x) # x는 샘플
        _, hidden = self.rnn(x) # 모든 타임스텝의 출력은 사용하지 않음, 은닉 상태만 활용
        outputs = self.dense(hidden[-1]) # 마지막 hidden state만 밀집층으로 전달
         return self.sigmoid(outputs)
         # 최종 은닉 상태의 크기는 (층 개수, 배치 크기, 뉴런 개수)이다.
model = IMDBRnn()
     

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import torch.optim as optim

criterion = nn.BCELoss() # Binary Cross Entropy 
optimizer = optim.Adam(model.parameters(), lr=2e-4) # lr: learning rate
     

train_hist = []
val_hist = []
patience = 2
best_loss = -1
early_stopping_counter = 0

epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets) # target과 차원을 맞추기 위해 squeeze()로 크기가 1인 차원 삭제
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    train_hist.append(train_loss)
    val_hist.append(val_loss)
    print(f"에포크:{epoch+1},",
          f"훈련 손실:{train_loss:.4f}, 검증 손실:{val_loss:.4f}")

    if best_loss == -1 or val_loss < best_loss:
        best_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_rnn_model.pt')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"{epoch+1}번째 에포크에서 조기 종료되었습니다.")
            break

import matplotlib.pyplot as plt

plt.plot(train_hist, label='train')
plt.plot(val_hist, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load('best_rnn_model.pt', weights_only=True))

model.eval()
corrects = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicts = outputs > 0.5
        corrects += (predicts.squeeze() == targets).sum().item()

accuracy = corrects / len(val_dataset)
print(f"검증 정확도: {accuracy:.4f}")   
# 검증 정확도: 0.7272
```
