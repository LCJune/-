## 핵심 키워드
> **선형회귀(linear regression)**: 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾는다. 특성이 하나면 직선 방정식이 된다.
> 선형회귀가 찾은 특성과 타깃 사이의 관계는 선형 방정식의 계수(*coef_*) 또는 가중치(*intercept_*)에 저장된다.
> 머신러닝에서 종종 가중치는 방정식의 기울기와 절편을 모두 의미하는 경우가 많다.

> **모델 파라미터**: 선형회귀가 찾은 가중치처럼 머신러닝이 모델이 특성에서 학습한 파라미터

> **다항회귀**: 다항식을 사용해 특성과 타깃 사이의 관계를 나타낸다. 함수는 비선형일 수 있지만 여전히 선형회귀로 표현할 수 있다.

 
## 핵심 패키지와 함수
### sckit-learn
> **LinearRegression**: 사이킷런의 선형회귀 클래스.
> *fit_intercept* 매개변수를 False로 지정하면 절편을 학습하지 않는다. 기본값은 True이다.
> 학습된 모델의 *coef_* 속성은 특성에 대한 계수를 포함한 배열이다. 즉, 이 배열의 크기는 특성의 개수와 같다.
> *intercept_* 속성에는 절편이 저장되어 있다.



## 코드 전문
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
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_input, train_target)

print(lr.predict([[50]]))
print(lr.coef_, lr.intercept_) #각각 1차함수의 기울기와 y절편

import matplotlib.pyplot as plt

plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))

print(lr.coef_, lr.intercept_)

point = np.arange(15, 50)

plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

plt.scatter(50,1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```
<img width="580" height="432" alt="image" src="https://github.com/user-attachments/assets/23bee4ab-5dca-4a31-80d8-0102dba18e5f" />
<img width="580" height="432" alt="image" src="https://github.com/user-attachments/assets/55c6f0a1-ee06-477e-8d31-b905b510b768" />

