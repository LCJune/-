## 핵심 키워드
> **로지스틱 회귀(logistic regression)**: 선형 방정식을 이용한 분류 알고리즘.
> 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 이용해 클래스 확률을 출력할 수 있다.  
> <img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/13643a92-7a88-435e-84f6-e9821c9fdbe7" />
><img width="300" height="120" alt="image" src="https://github.com/user-attachments/assets/1e3fa593-21cc-4381-8d02-3a0aff102b3c" />

> **다중 분류**: 타깃 클래스가 2개 이상인 분류 문제. 로지스틱 회귀는 다중 분류를 위해 소프트 맥스 함수를 이요해 클래스를 예측한다.

> **시그모이드 함수(sigmoid function)**: 로지스틱 함수라고도 부르며, 선형 방정식의 값을 0과 1 사이의 값으로 압축한다.
> 이진 분류(타깃 클래스가 2개인 분류 문제)를 위해 사용된다.

> **소프트맥스 함수(softmax function)**: 다중 분류에서 여러 선형 방정식의 출력 결괄르 정규화하여 합이 1이 되도록 만든다.
> 로지스틱 회귀 알고리즘은 다중 분류 시 모든 개별 클래스의 Z값(표준점수)를 산출하고, 그 중 가장 높은 값을 가진 클래스를 정답으로 예측한다.


## 핵심 패키지와 함수
### scikit-learn
> **LogisticRegression**: 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스이다.  
> *solver* 매개변수에서 사용할 알고리즘을 사용할 수 있다. sag, saga, newton-cholesky(대규모 데이터셋에 효율적) 등이 있다.  
> *penalty* 매개변수에서 L2규제 방식(ridge)과 L1규제 방식(lasso)를 선택할 수 있다. 기본값은 L2규제 방식을 의미하는 'l2'이다.
> *C* 매개변수에서 규제 강도를 제어한다. 기본값은 1이며, alpha와 다르게 수치가 작을수록 규제가 강해진다.

> **predict_proba()**: 예측 확률을 반환하는 메서드이다.
> 이진 분류의 경우에는 샘플마다 음성 클래스와 양성 클래스에 대한 확률을 반환한다. (두 클래스 중 앞의 클래스를 음성, 뒤의 클래스를 양성으로 지정한다.)
> 다중 분류의 경우에는 샘플마다 모든 클래스에 대한 확률을 반환한다.

> **decision_function()**: 모델이 학습한 선형 방정식의 출력을 반환한다.
> 이진 분류의 경우 양성 클래스의 확률이 반환된다. 이 값이 0보다 크면 양성 클래스, 작거나 같으면 음성 클래스로 예측한다.
> 다중 분류의 경우 각 클래스마다 선형 방정식을 계산한다. 가장 큰 값의 클래스가 예측 클래스가 된다.  
> 이때, decision_function()의 반환값을 각각 시그모이드, 소프트맥스 함수에 통과시키면 predict_proba() 함수가 출력하는 확률값이 나온다.


## 코드 전문
```python
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight','Length','Diagonal','Height','Width']]
fish_target = fish['Species']

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


"""
아래 구문에 쓰인 방식은 불리언 인덱싱(boolean indexing)이다.
numpy 배열의 index에 True, False와 같은 boolean 값으로 채워진 배열을 전달하면
True 값인 index의 요소만 추출해 반환한다. 이때, index의 길이는 배열의 길이와 같아야 한다.
"""
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

## 로지스틱 회귀 모델로 이진 분류 수행
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
decisions = lr.decision_function(train_bream_smelt[:5]) 
"""
로지스틱 회귀는 분류 시 클래스를 알파뱃 순으로 정렬해 사용한다.
위 이진 분류에서 클래스는 Bream, Smelt 두 가지이고, 두 번째 순서에 위치하는
Smelt가 양성 클래스로 분류된다.
"""

from scipy.special import expit
print(expit(decisions))

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 3))

from scipy.special import softmax
proba = softmax(lr.decision_function(test_scaled[:5]), axis=1)
print(np.round(proba, decimals=3))
```
