## 핵심 키워드
> **결정 트리(decision tree)**: 데이터의 특성을 질문(기준)으로 하여 좌, 우로 샘플들을 분류하는 알고리즘이다.
> 결정 트리는 여러개의 노드로 구성되어 있다. 노드는 훈련 데이터의 특성에 대한 테스트를 표현한다. 가장 위의 노드를 루트 노드, 맨 아래 끝의 노드를 리프 노드라 한다.
> 결정 트리는 이해하기 쉽고 설명에 용이한 구조를 가지고 있다.

> **불순도(impurity)**: 불순도는 트리의 각 노드가 가지고 있는 값으로, 여러 클래스가 섞여 있을수록 그 수치가 높다.
> 일반적으로 쓰이는 불순도는 지니(gini) 불순도와 엔트로피(entrophy) 불순도가 있다. 각각의 계산 식은 다음과 같다.
> <img width="388" height="112" alt="image" src="https://github.com/user-attachments/assets/06ca0143-c73b-4f49-b560-5db44d60f596" />
> <img width="197" height="79" alt="image" src="https://github.com/user-attachments/assets/6bb8024b-6722-48c7-a7e9-d257eca6f682" />

> **정보 이득(information gain)**: 부모 노드와 자식 노드의 불순도 차이. 계산 방법은 다음과 같다.  
> 부모의 불순도 - (왼쪽 노드 샘플 수 / 부모의 샘플 수) * 왼쪽 노드 불순도 - (오른쪽 노드 샘플 수 / 부모의 샘플 수) * 오른쪽 노드 불순도
> 결정 트리 알고리즘은 이 정보 이득을 최대가 되도록 데이터 분류룰 수행한다.

> **특성 중요도(feature imfortance)**: 결정 트리에 사용된 특성이 불순도를 감소시키는 데에 기여한 정도를 나타내는 값이다.
> 특성 중요도를 알면 어느 특성이 데이터 분류에 효과적인지 알 수 있다.


## 핵심 패키지와 함수
### pandas
> **info()**: 데이터 프레임의 요약된 정보를 출력한다. 인덱스와 컬럼 타입을 출력하고 널(null)이 아닌 개수, 메모리 사용량을 제공한다.  
> *verbose* 매개변수의 기본값 True를 False로 바꾸면 각 열에 대한 정보를 출력하지 않는다.

> **describe()**: 데이터프레임 열의 통계값을 제공한다. 수치형일 경우 최소, 최대, 평균, 표준편차와 사분위값 등이 출력된다.
> 문자열 같은 객체 타입의 열은 가장 자주 등장하는 값과 횟수 등이 출력된다.
> *percentiles* 매개변수에서 백분위수를 지정한다. 기본값은 [0.25, 0.5, 0.75]이다.


### sklearn
> **DecisionTreeClassifier**: 결정 트리 분류 클래스이다.  
> *criterion*: 불순도 계산 방식을 지정한다. 기본값은 지니 불순도를 의미하는 'gini'이고, 'entrophy'를 사용해 엔트로피 불순도를 사용할 수 있다.  
> *splitter*: 노드를 분할하는 전략을 선택한다. 기본값은 'best'로 정보 이득이 최대가 되도록 분할한다. 'random'이면 임의로 노드를 분할한다.  
> *max_depth*: 트리가 성장할 최대 깊이를 지정한다. 기본값은 None으로, 리프 노드가 순수하거나 min_sample_split보다 샘플 수가 적을 때까지 성장한다.  
> *min_sample_split*: 노드를 나누기 위한 최소 샘플 개수를 지정한다. 기본값은 2이다.  
> *max_features*: 최적의 분할을 위해 탐색할 특성의 개수를 지정한다. 기본값은 None으로, 모든 특성을 사용한다.  

> **plot_tree()**: 결정 트리 모델을 시각화 하는 함수이다. 첫 번째 매개변수로 결정 트리 모델 객체를 전달한다.
> *max_depth*: 나타낼 트리의 깊이를 지정한다. 기본값은 None으로, 모든 노드를 출력한다.
> *feature_names*: 특성의 이름을 지정한다.
> *filed*: True로 지정하면 타깃값에 따라 노드에 색을 채운다.


## 코드 전문
```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

wine.head()
wine.info()
wine.describe()

data = wine[['alcohol', 'sugar', 'pH']] # sklearn 라이브러리의 모델들은 모두 2차원 형태의 학습 데이터를 요구한다. (의도된 인터페이스 강제)
target = wine['class']

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
```
