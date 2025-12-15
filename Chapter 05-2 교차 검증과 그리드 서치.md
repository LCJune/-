## 핵심 키워드
> **검증 세트(validation set)**: 테스트 세트 대신 모델의 하이퍼 파라미터 튜닝 과정에서 훈련 세트의 일부를 떼어 사용하는 데이터이다.  
> 테스트 세트를 모델 테스트에 계속 사용하다가 모델이 테스트 세트에 맞게 조정되어 실전성이 떨어지는 문제를 막기 위해 사용되고 있다.  

> **교차 검증(cross validation)**: 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복하는 과정이다. 여러 번의 교차 검증을 통해 나온 점수들을 평균하여 최종 점수를 얻는다.  
> 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서는 모델을 훈련한다.  
> 이런 식으로 모든 폴드에 대해 검증 점수를 얻어 평균하는 방식을 K-폴드 교차 검증(k-fold cross validation)이라 한다.
> <img width="573" height="344" alt="image" src="https://github.com/user-attachments/assets/e268aa94-eec3-458d-a49d-d5a75898ced3" />

> **하이퍼 파라미터 튜닝**: 머신러닝이 학습하는 모델 파라미터와 달리, 사용자가 지정해야만 하는 파라미터를 하이퍼 파라미터(hyper parameter)라고 한다.  
> 모델의 성능을 높이기 위해 이 하이퍼 파라미터를 조정하는 과정을 하이퍼 파라미터 튜닝이라 한다.

> **그리드 서치(grid search)**: 하이퍼파라미터 탐색을 자동화 해주는 도구. 탐색할 매개변수를 나열하면 교차 검증을 수행해 가장 좋은 검증 점수의 매개변수 조합을 선택한다.  
> 마지막으로 해당 조합을 이용해 최종 모델을 훈련한다.

> **랜덤 서치(random search)**: 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달받아 지정된 횟수만큼 샘플링 하여 교차 검증을 수행해 매개변수 조합을 테스트한다.  
> 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어렵거나(연속된 매개변수일 때) 매개변수 조건이 너무 많아 그리드 서치가 어려울 때 사용한다.

## 핵심 패키지와 함수
### sklearn
> **cross_validate()**: 교차 검증을 수행하는 함수이다. 첫 번째 매개변수에 교차 검증을 수행할 모델 객체를 전달한다. 두 번째와 세 번째 매개변수에 특성과 타깃 데이터를 전달한다.  
> *scoring*: 매개변수 검증에 사용할 평가 지표를 지정한다. 기본값은 정확도를 의미하는 'accuracy', 회귀 모델은 결정계수를 의미하는 'r2'이다.  
> *cv*: 교차 검증 폴드 수나 분할기 객체를 지정할 수 있다. 기본값은 5이다. 회귀일 때는 KFold 클래스를 사용하고 분류일 때는 StratifiedKFold 클래스를 사용해 5-폴드 교차 검증을 수행한다.  
> *n_jobs*: 교차 검증 시 사용할 CPU 개수를 지정한다. 기본값은 1로, 하나의 코어를 사용한다. -1로 지정하면 시스템에 있는 모든 코어를 사용한다.  
> *return_train_score*: True로 지정하면 훈련 세트의 점수도 반환한다. 기본값은 False이다.

> **GridSearchCV**: 교차 검증으로 하이퍼파라미터 탐색을 수행한다. 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련한다.  
> 첫 번째 매개변수르 그리드 서치를 수행할 모델 객체를 전달한다. 두 번째 매개변수에는 탐색할 모데르이 매개변수와 값을 전달한다.  
> *scoring, cv, n_jobs, return_train_score* 매개변수는 **cross_validate()** 함수와 동일하다.

> **RandomizedSearchCV**: 교차 검증으로 랜덤한 하이퍼파라미터 탐색을 수행한다. 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련한다.  
> 첫 번째 매개변수로 그리드 서치를 수행할 모델 객체를 전달한다. 두 번째 매개변수에는 탐색할 모델의 매개변수와 확률 분포 객체를 전달한다.  
> *scoring, cv, n_jobs, return_train_score* 매개변수는 **cross_validate()** 함수와 동일하다.

## 코드 전문
```python
### 데이터 준비
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

### 교차 검증
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)

import numpy as np

print(np.mean(scores['test_score']))
0.855300214703487

from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
0.855300214703487
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

### 하이퍼 파라미터 튜닝 - 그리드 서치
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
"""
GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
             param_grid={'min_impurity_decrease': [0.0001, 0.0002, 0.0003,
                                                   0.0004, 0.0005]})
"""

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][gs.best_index_])

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
"""
GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
             param_grid={'max_depth': range(5, 20),
                         'min_impurity_decrease': array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
       0.0009]),
                         'min_samples_split': range(2, 100, 10)})
"""
print(gs.best_params_)
{'max_depth': 14, 'min_impurity_decrease': np.float64(0.0004), 'min_samples_split': 12}
print(np.max(gs.cv_results_['mean_test_score']))

### 랜덤 서치
from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)
array([6, 1, 0, 8, 1, 8, 8, 8, 8, 1])
np.unique(rgen.rvs(1000), return_counts=True)
"""
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
 array([116, 106,  96, 102, 102,  94,  97,  93, 101,  93]))
"""
ugen = uniform(0, 1)
ugen.rvs(10)
"""
array([0.18871786, 0.18333195, 0.22269547, 0.84036586, 0.81400407,
       0.66906917, 0.9170693 , 0.76208373, 0.80087916, 0.30102417])
"""
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }
from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter=100, n_jobs=-1, random_state=42)
rs.fit(train_input, train_target)
print(rs.best_params_)
"""
{'max_depth': 39, 'min_impurity_decrease': np.float64(0.00034102546602601173), 'min_samples_leaf': 7, 'min_samples_split': 13}
"""
print(np.max(rs.cv_results_['mean_test_score']))
#output: 0.8695428296438884

dt = rs.best_estimator_
print(dt.score(test_input, test_target))
#output: 0.86
```
