## 핵심 키워드
> **정형 데이터(structured data)**: 구조를 갖춘 데이터를 이르는 말. 행과 열로 각 샘플과 특성을 나타내는 데이터를 정형 데이터의 예시로 들 수 있다.

> **비정형 데이터(unstructured data)**: 특정한 구조가 없어, 데이터베이스나 엑셀로 표현하기 어렵다. 텍스트 데이터, 디지털 사진, 디지털 음악 등이 예시이다.

> **앙상블 학습(ensemble learning)**: 여러 개의 개별 모델(약한 학습기, base learner)을 결합하여 단일 모델보다 더 높은 예측 성능과 일반화 능력을 얻는 머신러닝 기법.
> 이러한 방식은 분산을 감소시켜 과대적합에 강하고, 편향을 감소시켜 단순 모델의 한계를 보완한다. 또한 데이터 변동에 덜 민감하여 안정성이 높다.

> **랜덤 포레스트(random forest)**: 앙상블 학습의 대표적인 모델 중 하나. 샘플 데이터 중에서 중복을 허용하여 데이터를 뽑은 부트스트랩 샘플을 이용해 여러 개의 결정 트리를 학습시킨다.
> 결정 트리의 각 노드를 분할할 때 일부 특성을 무작위로 고른 다음, 최선의 분할(정보 이득 최대)을 찾는다. RandomForestClassifier의 경우, 기본적으로 전체 특성 개수의 제곱근만큼의 특성을 선택한다.
> 이런 방식으로 트리를 훈련한 후, 분류의 경우 각 트리의 클래스별 확률을 평균해 가장 높은 클래스를 예측으로 삼는다. 회귀일 때는 단순히 각 트리의 예측을 평균한다.

> **엑스트라 트리(extra tree)**: 랜덤 포레스트와 유사한 앙상블 알고리즘. 결정 데이터를 만들 때 전체 훈련 세트를 사용한다.
> 엑스트라 트리의 각 결정 트리는 노드 분할 시 일부 특성을 랜덤으로 선택하여 분할하지만, 최선의 분할을 찾지 않고 무작위로 분할한다.
> 즉, 엑스트라 트리가 사용하는 결정 트리는 DecisionTreeClassifier의 *splitter* 매개변수 값을 'random'으로 준 결정 트리이다.

> **그레디언트 부스팅(gradient boosting)**  
> : 깊이가 얕은 결정 트리를 순차적으로 추가하여 이전 모델의 오차를 보정하는 앙상블 학습 방법이다. 전체 모델은 손실 함수를 최소화하는 방향으로 경사 하강 방식으로 업데이트된다.  
> 초기에는 상수 함수 𝐹0를 설정하고, 이후 각 단계에서는 현재 예측값에서 계산된 손실함수의 음의 gradient 값(pseudo-residual)을 입력 특성 공간에서 함수로 근사하는 회귀 트리를 학습한다.  
> 이는 모델이 현재 예측에서 틀린 방향과 크기를 학습하는 과정이다.  
> 각 트리는 샘플의 특성 데이터를 입력으로 사용하는 깊이가 매우 얕은 회귀 트리이며, 분할 시 pseudo-residual에 대한  
> Δ = SSEparent − (SSEleft + SSEright) 가 최대가 되도록 분할한다.  
> Δ가 최대가 되면 각 리프 노드 내의 pseudo-residual 분산이 감소하며, 이는 동일한 상수 보정값을 적용해도 해당 노드에 속한 샘플들의 손실이 효과적으로 감소함을 의미한다.  
> 이 과정은 실제 클래스와 무관하게 이루어지며, 리프 노드의 값은 해당 노드 내 샘플들의 손실을 최소화하도록 계산된다.  
> 전체 모델을 수식으로 표현하면 다음과 같다.  
> <img width="264" height="66" alt="image" src="https://github.com/user-attachments/assets/e0d4ad1b-56eb-42f7-b5b2-d2abe4d57c08" />  
> F0(x): 초기 상수 
> η: learning rate 
> 각 ℎ𝑚(𝑥): 입력 의존적 보정 함수  
> 그레디언트 부스팅의 속도를 개선한 버전인 히스토그램 기반 그레디언트 부스팅이 안정적인 결과와 높은 성능으로 인기가 높다.


## 코드 전문
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

### Randomized Forest
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
"""
cross_validate() 함수는 dictionary를 반환한다.
key는 'fit_time', 'score_time', 'test_score', return_train_score를 True로 지정 시 'train_score', 'train_accuracy', 'train_f1'이 추가된다.
"""
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
rf.fit(train_input, train_target)
print(rf.feature_importances_)
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

rf.fit(train_input, train_target)
print(rf.oob_score_)

### Extra Tree
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

et.fit(train_input, train_target)
print(et.feature_importances_)

### gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2,
                                random_state=42)
scores = cross_validate(gb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

gb.fit(train_input, train_target)
print(gb.feature_importances_)

### hist gradient boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
[0.08876275 0.23438522 0.08027708]
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)
hgb.score(test_input, test_target)

# XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)
xgb._estimator_type = "classifier"
scores = cross_validate(xgb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
LightGBM

### LightGBM
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

```
