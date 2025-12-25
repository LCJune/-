## 핵심 키워드
**인공 신경망(Artificial Neural Network)**  
생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘이다. 기존의 머신러닝 알고리즘으로 다루기 어려운  
이미지, 음성, 텍스트 분야에서 뛰어난 성능을 발휘한다. 인공 신경망 알고리즘을 종종 딥러닝이라고도 부른다.  


**케라스(KERAS)**  
대표적인 딥러닝 라이브러리이다. 텐서플로(TensorFlow)의 고수준 API이다.  
딥러닝 라이브러리는 머신러닝과 차별되게 GPU 연산을 사용해 모델을 훈련시킨다.
GPU는 행렬과 벡터 연산에 최적화 되어 있기 때문에 곱셈과 덧셈이 많이 수행되는 인공지능 훈련에 좋은 성능을 보인다.  
케라스 라이브러리는 GPU 연산을 직접 수행하지 않고 백엔드(backend)에 맡긴다.  
케라스의 백엔드로 쓰이는 라이브러리로는 최신버전인 케라스3.0 기준으로 텐서플로, 파이토치(Pytorch), 잭스(Jax)가 있다.

**층(layer)**  
인공 신경망 모델의 구조는 크게 3가지 층으로 나뉜다.  
> <img width="709" height="437" alt="image" src="https://github.com/user-attachments/assets/bcf130ff-f06f-40af-8d22-a421b94c5329" />  

* 입력층(Input Layer): 원시 특성(feature)를 모델 내부로 전달한다.  
  별다른 계산을 수행하지 않고 데이터의 형태만 정의한다.  
  입력층의 노드 개수는 특성의 수와 같다.  

* 은닉층(Hidden Layer): 인공 신경망에서 입력층과 출력층 사이에 있는 모든 층을 의미한다. 신경망의 핵심 연산부이며, 여러개가 존재할 수 있다.  
  각 은닉층에서 수행되는 연산은 다음과 같다.  
  <img width="569" height="751" alt="image" src="https://github.com/user-attachments/assets/ac796e66-7370-4a33-97d0-7c654c8704b5" />  
  은닉층은 원시 특성 데이터에서 고수준의 특성을 추출하고 입력 데이터의 표현을 추상화 시킨다.  
  이미지 데이터를 예시로 들면 다음과 같다.  
  * 이미지  
  * 1층: 엣지, 선  
  * 2층: 모서리, 간단한 패턴  
  * 3층: 객체의 부분
  * 수치 데이터
  * 변수 간 상호작용 학습
  * 비선형 조합 생성
    
* 출력층(Output Layer): 은닉층에서 생성된 표현을 문제의 목적에 맞는 출력 형태로 변환한다.  
  구성은 문제의 유형에 따라 달라진다.  
  > <img width="477" height="687" alt="image" src="https://github.com/user-attachments/assets/3244fc03-7c49-4282-a1ab-9dab93b93794" />  

* 밀집층(Dense Layer): 이전 층의 모든 뉴런과 완전히 연결된(fully connected) 층  
  은닉층과 혼동하기 쉽지만, 둘은 분류 기준이 다르다. 은닉층은 위치/역할로서의 분류라면 밀집층은 연결 관계로서의 분류이다.
  > <img width="517" height="413" alt="image" src="https://github.com/user-attachments/assets/ff0c5b8c-aa46-42cf-9012-fedb3aeb2546" />
  > σ: 활성화 함수
  
  
## 핵심 패키지와 함수
### KERAS
* **Input()**   
  입력층을 구성하기 위한 함수이다. *shape* 매개변수에 입력의 크기를 튜플로 지정한다.  

* **Dense()**    
  밀집층을 구성하기 위한 함수이다. 첫 번째 매개변수에는 뉴런의 개수를 지정한다.  
  *activation*: 사용할 활성화 함수를 지정한다. 대표적으로 'softmax', 'sigmoid'가 있다.  
   아무것도 지정하지 않으면 활성화 함수를 사용하지 않는다.(선형 변환만 수행)  

* **Sequential()**    
  케라스에서 신경망 모델을 만드는 클래스이다.  
  이 클래스의 객체를 생성할 때 신경망 모델에 추가할 층을 파이썬 리스트로 전달한다.  

* **compile()**    
  모델 객체를 만든 후 훈련하기 전에 사용할 손실 함수와 측정 지표 등을 지정하는 메서드이다.
  *optimizer*: 옵티마이저를 지정한다. 'sgd', 'rmsdrop', 'adam'등이 있다.
  
  *loss*: 손실함수를 지정한다.  
   이진 분류일 경우 'binary_crossentropy',  
   다중 분류일 경우 'categorical_crossentropy',  
   다중 분류면서 클래스 레이블이 정수일 경우 'sparse_categorical_crossentropy'  
   회귀 모델일 경우 'mean_square_error' 등으로 지정할 수 있다.  

  *metrics*: 훈련 과정에서 측정하고 싶은 지표를 리스트로 전달한다.  
   기본적으로 'loss(손실)'이 포함 돼 있고, 'accuracy(정확도)' 등을 포함할 수 있다.  
   회귀의 경우 'mse(mean squared error)', 'mae(mean absolute error)' 등을 전달할 수 있다.  

* **fit()**  
  모델을 훈련하는 메서드이다.
  첫 번째와 두 번째 입력과 타깃 데이터를 전달한다.  
  *epoch*: 전체 데이터에 대해 반복할 에포크 횟수를 지정한다.  

* **evaluate()**
  모델 성능을 평가하는 메서드이다.
  첫 번쨰와 두 번째 매개변수에 입력과 타깃 데이터를 전달한다.
  compile() 메서드에서 *loss* 매개변수에 지정한 손실함수의 값과 *metrics* 매개벼수에서 지정한 측정 지표를 출력한다.


## 코드 전문
```python
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다.
import keras
import tensorflow as tf

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

## 패션 MNIST
import keras

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
# (60000, 28, 28) (60000,)
print(test_input.shape, test_target.shape)
# (10000, 28, 28) (10000,)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print(train_target[:10])
# [9 0 0 3 0 2 7 2 5 5]
import numpy as np

print(np.unique(train_target, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))

## 로지스틱 회귀로 패션 아이템 분류하기
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)
# (60000, 784)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

## 인공 신경망 - 텐서플로와 케라스
import tensorflow as tf
import keras
keras.config.backend()
'tensorflow'
import os
os.environ["KERAS_BACKEND"] = "torch"   # 또는 "jax"

## 인공 신경망으로 모델 만들기
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, train_target.shape)
# (48000, 784) (48000,)
print(val_scaled.shape, val_target.shape)
# (12000, 784) (12000,)
inputs = keras.layers.Input(shape=(784,))
dense = keras.layers.Dense(10, activation='softmax')
model = keras.Sequential([inputs, dense])

## 인공 신경망으로 패션 아이템 분류하기
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(train_target[:10])
# [7 3 5 8 6 9 3 3 9 9]
model.fit(train_scaled, train_target, epochs=5)
"""
Epoch 1/5
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 8s 3ms/step - accuracy: 0.7370 - loss: 0.7853
Epoch 2/5
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 7s 3ms/step - accuracy: 0.8346 - loss: 0.4845
Epoch 3/5
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.8452 - loss: 0.4564
Epoch 4/5
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - accuracy: 0.8504 - loss: 0.4425
Epoch 5/5
1500/1500 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.8537 - loss: 0.4337
<keras.src.callbacks.history.History at 0x7ed658e64050>  << fit 메서드는 history 객체를 반환한다.
"""
model.evaluate(val_scaled, val_target)
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8462 - loss: 0.4364
# [0.4444445073604584, 0.8458333611488342]
```

    
  
  

