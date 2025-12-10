## 핵심 키워드
> **점진적 학습**: 훈련 데이터에 새로운 데이터가 추가되었을 때, 모델을 처음부터 다시 학습하지 않고
> 새로운 데이터에 대해서 조금씩 훈련하는 훈련 방식

> **손실 함수(loss function)**: 머신러닝/딥러닝 모델이 예측한 값과 실제 정답 값 사이의 '차이' 또는 '오차'를 수치로 측정하는 함수. 손실 함수의 값이 작을수록 모델의 성능이 뛰어나다.
> 분류 모델에서 대표적으로 쓰이는 손실 함수로는 logistic(binary cross-entropy), softmax, hinge, categorical cross-entrophy가 있다. 회귀 모델에선 mean squared error 함수가 널리 쓰인다.

## 확률적 경사 하강법(stochastic gradient descent, SGD)**
> 경사 하강법이란, 손실 함수의 기울기를 따라 내려가며 파라미터를 조정해 손실 함수의 값을 최대한 작게 최적화하며 모델의 성능을 높이는 알고리즘이다.  
> 이때 데이터를 무작위로 하나 뽑아서 쓰면 확률적, 여러개를 뽑아서 쓰면 mini batch, 모든 데이터를 한 번에 쓰면 batch 경사 하강법이다.
> SGD 방식으로 훈련할 때, 훈련 세트를 한 번 모두 사용하는 과정을 **에포크(epoch)**라고 한다. 보통 SGD는 수십, 수백 번 에포크를 반복하며 학습한다.

> 손실 함수의 경사를 하강한다는 것은, 손실 함수가 모델의 파라미터에 대해 미분 가능할 때,  
> 현재 파라미터 θ에서 손실 함수 L(θ)의 기울기(gradient) ∇θL을 계산하고,  
> 그 기울기의 반대 방향(−∇θL)으로 파라미터를 조금 이동시키는 업데이트를 반복함으로써,  
> 손실 함수가 낮아지는 방향으로 파라미터를 점진적으로 조정하여  
> 손실 함수의 지역 최소점(혹은 전역 최소점)에 도달하는 방법을 말한다.  

>에포크를 반복 할수록 모델은 데이터 전반에 걸쳐 안정적인 성능을 내도록 최적화된다.
>시간이 지나면서 업데이트 크기가 작아지고 w(weight, 가중치), b(bias, 편향)가 데이터 전체에 맞는 균형점으로 수렴한다.

## 핵심 패키지와 함수
**SGDClassifier**: 확률적 경사 하강법을 사용한 분류 모델을 만드는 클래스이다.
> *loss* 매개변수로 손실 함수의 종류를 지정할 수 있다. 기본값은 서포트 벡터 머신(Support Vactor Machine, SVM)을 위한 hinge loss이다. 로지스틱 회귀를 위해서는 'log_loss'로 지정한다.  
> *penalty* 매개변수로 규제의 종류를 지정할 수 있다. 기본값은 L2(ridge)규제를 위한 'l2'이다. L1(lasso)규제를 적용하려면 'l1'으로 지정한다.  
> *alpha* 매개변수로 규제의 강도를 지정할 수 있다. 기본값은 0.0001이다.  
> *max_iter* 매개변수로 에포크 횟수를 지정한다. 기본값은 1000이다.  
> *tol* 함수는 반복을 멈출 조건이다. *n_iter_no_change* 매개변수에서 지정한 에포크 동안 손실이 tol만큼 줄어들지 않으면 알고리즘이 중단된다. *tol* 기본값은 0.001이고, *n_iter_no_change*의 기본값은 5이다.

**SGDRegressor**: 확률적 경사 하강법을 이용한 회귀 모델을 만드는 클래스이다.
> *loss* 매개변수에서 손실함수를 지정한다. 기본값은 제곱 오차를 나타내는 'squared_loss'이다.
> SGDClassifier와 대부분의 매개변수의 쓰임을 공유한다.

## 코드 전문
```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']] # [[]]으로 여러개의 열을 지정해 data frame으로 만들 수 있다.
fish_target = fish['Species']

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state = 42
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log_loss', max_iter = 10, random_state = 42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

import numpy as np
sc = SGDClassifier(loss = 'log_loss', random_state = 42)
train_score = []
test_score = []

classes = np.unique(train_target)
print(classes)

"""
모델의 최초 훈련에 partial_fit()이 호출될 시, partial_fit의 classes 매개변수에 훈련 데이터의 모든 클래스를 전달해주어야 한다.
부분 데이터의 학습을 진행하므로, 해당 데이터에 모든 클래스가 포함되어 있을 것이라는 보장이 없기 때문이다.
최초 호출 시에 모든 클래스를 전달해 주었다면 이후부터 생략 가능하나, 새로운 클래스가 포함된 데이터를 학습할 시에는 다시 classes에
모든 클래스 이름을 전달해야 한다.
"""
for _ in range(0, 300):
  sc.partial_fit(train_scaled, train_target, classes=classes)
  train_score.append(sc.score(train_scaled, train_target))
  test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel = 'Epoch'
plt.ylabel = 'Accuracy'
plt.show()

sc = SGDClassifier(loss = 'log_loss', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc = SGDClassifier(loss = 'hinge', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
```
