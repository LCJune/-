## 핵심 키워드
> **다중회귀(multiple regression)**: 여러 개의 특성을 활용하는 회귀모델. 특징이 많으면 선형 모델은 강력한 성능을 발휘한다.

> **특성 공학(feature engineerring)**: 데이터의 기존 특성을 이용해 새로운 특성을 뽑아내는 작업

>**규제(regulation)**: 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 않도록 조절하는 작업.
>즉, 모델이 훈련 세트에 과대적합 되지 않도록 만드는 것이다.

>**릿지(ridge)**: 규제가 있는 선형 회귀 모델 중 하나이며 선형 모델의 게수를 작게 만들어 과대적합을 완화시킨다.
>효과가 좋아 비교적 널리 사용된다.

>**라쏘(lasso)**: 또 다른 규제가 있는 선형 회귀 모델 중 하나. 릿지와 달리 계수를 아예 0으로 만들 수도 있다.
>이 때문에 효과적인 특성을 골라내는 데에 사용되기도 한다.

>**하이퍼파라미터(hyperparameter)**: 머신러닝 모델이 학습할 수 없고 사람이 직접 정해주어야 하는 매개변수


## 핵심 패키지와 함수
### pandas
> **read_csv()**: CSV 파일을 로컬 컴퓨터나 인터넷에서 읽어 판다스 데이터프레임으로 변환하는 함수이다.  
> 자주 사용하는 매개변수는 다음과 같다.  
> *sep: 파일의 구분자를 지정한다. 기본값은 콤마(,)이다.
> *header: 데이터프레임의 열 이름으로 사용할 CSV 파일의 행 번호를 지정한다. 기본값은 첫 번째 행이다.  
> *skiprows: 파일에서 읽기 전에 건너뛸 행의 개수를 지정한다.
> *nrows: 파일에서 읽을 행의 개수를 지정한다.

### sckit-learn
**PolynomialFeatures**: 주어진 특성을 조합해 새로운 특성을 만든다. 매개변수는 다음과 같다.  
>*degree: 최고차수를 지정한다. 기본값은 2이다.
>*interaction_only: True이면 거듭제곱 항은 제외되고 특성 간의 곱셈 항만 추가된다. 기본값은 False이다.  
>*include_bias: False이면 절편을 위한 특성을 추가하지 않는다. 기본값은 True이다.  

**Ridge**: 규제가 있는 회귀 알고리즘인 릿지 회귀 모델을 훈련한다. 매개변수는 다음과 같다.
>*alpha: 규제의 강도를 지정한다. 수치가 높을수록 규제 강도가 높아진다. 기본값은 1이다.  
>*sovler: 최적의 모델을 찾기 위한 방법을 지정할 수 있다. 기본값은 'auto'이며, 데이터에 따라 자동으로 선택된다.
>대표적으로 sag(확률적 경사 하강법)과 그 개선 버전인 saga가 있다.
>*random_state: solver가 'sag'나 'saga'일 때 넘파이 난수 시드값을 지정할 수 있다.  

**Lasso**: 규제가 있는 회귀 알고리즘인 라쏘 회귀 모델을 훈련한다. 좌표 하강법을 사용해 최적의 모델을 찾는다.  
매개변수는 ridge와 대부분 비슷하다. *max_iter* 매개변수로 알고리즘의 수행 반복 횟수를 지정한다. 기본값은 1000이다.  


## 코드 전문
```python
import pandas as pd
perch_full = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full.head()

import numpy as np

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
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

##데이터 변형(속성 추가)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

## 규제 모델 훈련 전 데이터 전처리(정규화)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

##릿지 모델 훈련
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_poly, train_target)
print(ridge.score(train_poly, train_target))

import matplotlib.pyplot as plt

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha, max_iter = 20000)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
plt.plot(alpha_list, train_score)
plt.plot(alpha_list, test_score)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()


##라쏘 모델 훈련
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_poly, train_target)
print(lasso.score(train_poly, train_target))

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=20000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(alpha_list, train_score)
plt.plot(alpha_list, test_score)
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```
<img width="576" height="437" alt="image" src="https://github.com/user-attachments/assets/17b82c14-affc-4382-80ce-8789815c8036" />
<img width="576" height="437" alt="image" src="https://github.com/user-attachments/assets/60c41f0b-80d7-4532-88d5-df4899ef6f73" />




