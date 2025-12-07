## 핵심 키워드
> **회귀(regression)**: 훈련 데이터를 바탕으로 임의의 새로운 샘플값을 예측하는 문제. 따라서 타깃값 역시 임의의 수치이다.

> **K-최근접 이웃 회귀**: K-최근접 이웃 알고리즘을 사용해 회귀문제를 푼다.
> 가장 가까운 K개 이웃의 샘플들의 타깃값을 평균하여 예측으로 삼는다.

> **결정계수(R²)**: 대표적인 회귀 문제의 성능 측정 도구. R² = 1 - {Σ(타깃값 - 예측)² / Σ(타깃값 - 평균)²}이다.
> 예측이 타깃에 아주 가까워지면 결정계수의 값이 1에 가까워지기에, 높을수록 좋다.

> **과대적합(overfitting)**: 모델의 훈련 세트 성능이 테스트 세트 성능보다 훨씬 높을 때 일어난다.
> 모델이 훈련 세트에 너무 집착해서 데이터에 내재된 거시적 패턴을 감지하지 못하는 경우이다.
> 이 경우, 모델을 조금 더 단순하게 만들어 데이터 전반의 일반적인 패턴을 감지하도록 해야한다.

> **과소적합(underfitting)**: 모델의 훈련 세트와 테스트 세트 성능이 모두 낮거나, 테스트 세트 성능이 더 높을 때 일어난다.
> 일반적으로 모델 학습 시 훈련 세트가 전체 데이터를 대표한다고 기대하기에, 훈련 세트 성능이 좋게 나와야 한다.
> 이 경우, 모델을 조금 더 복잡하게 만들어 훈련 세트의 국지적인 패턴에 좀 더 민감하도록 해야한다.


## 핵심 패키지와 함수
### sckit-learn
> **KNeighborsRegressor**: K-최근접 이웃 회귀 모델을 만드는 사이킷런 클래스. *n_neighbors* 매개변수로 이웃 개수를 정한다.
> 나머지 매개변수는 KNeighborsClassifier와 비슷하다.
>
> **mean_absolute_error()**: sklearn.metrics 패키지에 있다. 회귀 모델의 평균 절댓값 오차를 계산한다.  
> 첫 번쨰 매개변수는 타깃, 두 번째 매개변수는 예측값을 전달한다.  
> 이와 비슷한 함수로는 평균 제곱 오차를 계산하는 **mean_squared_error()**가 있다.

### numpy
> **reshape()**: 배열의 크기를 바꾸는 메서드. 바꾸고자 하는 배열의 크기를 매개변수로 전달한다.
> 바꾸기 전후의 원소 개수는 동일해야 한다. 매개변수로 -1을 전달할 경우, 다른 차원을 채우고 남은 원소에 맞게 차원을 정하라는 의미이다.


```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1) 

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

print(knr.score(test_input, test_target))

from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target, test_prediction)
print(mae)
#output: 19.157142857142862 -> 타깃값과 평균 19g 정도의 오차

print(knr.score(train_input,train_target))

knr.n_neighbors = 3 #과소적합 문제 해결을 위해 모델을 복잡하게 만듦

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

knr = KNeighborsRegressor()

x = np.arange(5, 45).reshape(-1,1)

for n in [1, 5, 10]:
  knr.n_neighbors = n
  knr.fit(train_input, train_target)
  prediction = knr.predict(x) # 농어의 길이 x(5 ~ 45)에 대한 무게(타깃값) 예측

  plt.scatter(train_input, train_target)
  plt.plot(x, prediction) #x, y값에 따라 선 그래프를 그리는 함수이다.
  plt.title('n_neighbors = {}'.format(n))
  plt.xlabel('length')
  plt.ylabel('weight')
  plt.show()
```
<img width="290" height="227" alt="image" src="https://github.com/user-attachments/assets/a003aa25-bd41-4a7f-8ab7-ad807b3fa56c" />
<img width="290" height="227" alt="image" src="https://github.com/user-attachments/assets/8c7a8571-764c-478e-9d75-925187eea437" />
<img width="290" height="227" alt="image" src="https://github.com/user-attachments/assets/2b58f921-f490-452a-8feb-0d2dac4d8e92" />



