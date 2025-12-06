## 핵심 키워드
> **데이터 전처리(data preprocessing)**: 데이터의 특성값들을 일정한 기준으로 맞추는 과정이다. 데이터 특성들의 스케일(scale)을 맞춘다고도 한다.

> **표준점수(standard score)**: 각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다.
> 즉, 데이터가 원점으로부터 몇 표준편차만큼 떨어져 있는지를 나타내는 값이다.

> **브로드캐스팅(broadcasting)**: 크기가 다른 넘파이 배열에서 자동으로 사칙연산을 모든 행이나 열로 확장하여 수행하는 기능

## 핵심 패키지와 함수
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###scikit-learn(sklearn)
> **train_test_split()**: 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수이다.
> 테스트 세트로 나눌 비율은 *test_size* 매개변수에서 지정할 수 있으며 기본값은 0.25(25%)이다.
>
> *shuffle* 매개변수로 훈련 세트와 테스트 세트로 나누기 전에 무작위로 섞을지 여부를 결정할 수 있다. 기본값은 True이다.
> *stratify* 매개변수에 클래스 레이블이 담긴 배열(일반적으로 타깃 데이터)을 전달하면 클래스 비율에 맞게 훈련 세트와 테스트 세트를 나눈다.

> **kneighbors()**: k-최근접 이웃 객체의 메서드이다. 이 메서드는 입력한 데이터에 가장 가까운 이웃을 찾아 거리와 이웃 샘플의 인덱스를 반환한다.
> 기본적으로 이웃의 개수는 KNeighborsClassifier 클래스의 객체를 생성할 때 지정한 개수를 사용하지만, *n_neighbors* 매개변수에서 다르게 지정할 수도 있다.
>
> *return_distance* 매개변수를 False로 지정하면 이웃 샘플의 인덱스만 반환하고 거리는 반환하지 않는다. 이 매개변수의 기본값은 True이다.
```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np 

fish_data = np.column_stack(  (fish_length, fish_weight)  ) """column_stack()은 매개변수로 받은 배열들의 같은 인덱스의
                                                               원소를 한 행에 위치하도록 하여 새로운 배열을 만든다."""

fish_target = np.concatenate( (np.ones(35), np.zeros(14)) ) """concatenate()는 매개변수로 받은 배열들을 1차원으로 연결한다.
                                                               ones()와 zeros()는 매개변수로 받은 정수의 개수만큼 각각 1, 0
                                                               으로 이루어진 배열을 생성한다."""

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42) # stratify - 클래스 비율에 맞게 테스트 데이터 설정, random_state - 랜덤 시드 정수 설정

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

kn.score(test_input, test_target)

"""
모델에 데이터를 학습시킬 때, 각 데이터의 수치적 단위(scale)를 맞추는 과정이 필요하다.
이 과정을 거치지 않으면 알고리즘이 데이터를 사용하는 과정에서 올바르지 않은 결과를 출력할 수 있기 때문이다. 
본 코드에 사용된 방법은 가장 보편적인 방법인 '표준화(standardization)'이다.
"""
mean = np.mean(train_input, axis = 0) #훈련 데이터의 평균
std = np.std(train_input, axis = 0) #훈련 데이터의 표준편차
train_scaled = (train_input - mean) / std # 데이터 전처리 - 표준화(standardization) 과정
new = ([25,150] - mean) / std #새로운 표본 또한 표준화한 후에 추가

import matplotlib.pyplot as plt
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

'''
테스트 세트를 전처리 할 때에도 훈련 세트를 전처리 할 때 사용한 방법과 동일한 방법으로 사용해야 한다.
즉, 훈련 세트의 통계값을 기반으로 표준화 해야한다.
만약 훈련 세트와 테스트 세트를 모두 합친 전체 데이터의 통계값으로 데이터 표준화를 수행하면 모델에 테스트
세트의 정보가 유출된 셈이기 때문이다. 이를 정보 누설(Information Leak) 또는 데이터 누설(Data Leak)이라고 한다.
'''
test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
kn.score(test_scaled, test_target)

print(kn.predict([new]))
#output: 1

distances, indexes = kn.kneighbors([new]) #new로부터 가까운 5개 이웃의 거리와 인덱스
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D') #가장 가까운 이웃 5개를 마름모로 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="585" height="432" alt="Image" src="https://github.com/user-attachments/assets/e44dfc37-8a8e-4309-8635-354bd68ea170" />
