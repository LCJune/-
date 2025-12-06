## 핵심 키워드
> **지도 학습(supervised learning)**: 입력과 타깃을 전달하여 모델을 훈련한 다음 새로운 데이터를 예측하는 데 활용.
K-최근접 이웃 알고리즘 또한 지도 학습 알고리즘이다. 지도 학습에서는 데이터와 정답을 **입력(input)**과 **타깃(target)**으로 분류하며, 이들을 합쳐 **훈련 데이터(training data)**라고 부른다.
지도 학습의 알고리즘은 정답을 맞히는 것을 학습한다. ex)데이터 분류

> **비지도 학습(unsupervised learning)**: 타깃 데이터 없이 입력만을 사용하여 학습한다. 학습에 정답을 사용하지 않으므로 무언가를 맞힐 수 없다.  
데이터의 파악과 변형, 패턴 탐색에 쓰인다.  

> **훈련 세트(training set)**: 모델을 훈련할 때 사용하는 데이터이다. 보통 훈련 세트가 클수록 학습 성과가 좋기 때문에, 테스트 데이터를 제외한 모든 데이터를 사용한다.  

> **테스트 세트(test set)**: 모델의 성능을 평가할 때 사용하는 데이터이다. 전체 데이터에서 20% ~ 30% 정도를 테스트 세트로 사용하는 경우가 많다. 전체 데이터가 아주 크다면 1%만 덜어내도 충분할 수 있다.

> **샘플(sample)**: 사용하고자 하는 특성을 모두 가진 하나의 데이터

> **샘플링 편향(sampling bias)**: 훈련 세트와 테스트 세트가 잘못 만들어져 전체 데이터를 대표하지 못하는 현상

## 핵심 패키지와 함수
 ### numpy
: python의 대표적인 배열 라이브러리이다. 고차원의 배열을 손쉽게 만들고 조작할 수 있는 간편한 도구를 많이 제공한다.

>**seed()**: 넘파이에서 난수를 생성하기 위한 정수 초깃값을 지정하는 함수. 초깃값이 같으면 동일한 난수를 생성한다.
>
>**arange()**: 일정한 간격의 정수 또는 실수 배열을 만들며, 기본 간격은 1이다. 매개변수가 하나이면 종료숫자, 둘이면 시작, 종료 숫자를 의미하며 매개변수가 3개면 마지막 매개변수가 간격을 나타낸다.
>
>**shuffle()**: 주어진 배열을 랜덤하게 섞는다. seed() 함수의 영향을 받는다.

```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l,w] for l, w in zip(fish_length, fish_weight)
fish_target = [1]*35 + [0]*14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

import numpy as np #numpy 함수를 np로 줄여서 사용

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42) #numpy의 random 함수의 초기 정수 시드를 42로 지정
index = np.arrange(49) # 0~48의 정수가 순서대로 담긴 index 배열 생성
"""
np.random.shuffle(index) #index 배열을 랜덤으로 섞음

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

#전체 49개 데이터 중 14개를 테스트 데이터로 지정
test_input = input_arr[index[35:]] 
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plx.ylabel('weight')
plt.show()

kn.fit(train_input, train_target) #사이킷런은 데이터(배열)이 행: 샘플, 열: 특성 으로 구성되어 있을 것으로 기대한다.

kn.score(test_input, test_target)
#output: 1.0 (정확도 100%)

kn.predict(test_input)
#output: array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])

print(test_target)
#output: array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0])
```
<img width="580" height="432" alt="image" src="https://github.com/user-attachments/assets/c9a17f89-fd8b-4905-b3eb-850c6a2a293d" />
