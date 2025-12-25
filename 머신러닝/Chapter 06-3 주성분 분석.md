## 핵심 키워드
* **차원 축소**: 원본 데이터의 특성을 적은 수의 새로운 특성으로 변환하는 비지도 학습의 일종.  
  차원 축소를 하면 저장 공간을 줄이고 시각화하기 쉬우며, 다른 알고리즘의 성능을 높일 수도 있다.  
* **주성분 분석**: 차원 축소 알고리즘의 하나로, 데이터에서 분산이 가장 큰 방향을 찾는 방법이다.
  해당 방향을 가리키는 벡터를 주성분이라고 부른다. 원본 데이터를 주성분에 투영하면 새로운 특성을 만들 수 있다.  
  일반적으로 주성분은 원본 데이터에 있는 특성 개수보다 작다.
* **설명된 분산(explaned variance)**: 주성분 분석에서 각 주성분이 얼마나 원본 데이터의 분산을 잘 나타내는지 기록한 것.

## 주성분 분석(Principal Component Analysis, PCA)
> 대표적인 차원축소 알고리즘의 하나로, 데이터에 있는 분산이 가장 큰 방향 벡터(주성분)를 찾는다.  
> 분산이 크다는 것은 “데이터가 어떻게 퍼져 있는지”, “어디에서 많이 변하는지”, “어디는 거의 변하지 않는지”  
> 등을 적은 정보 손실로 담아낼 수 있다는 것을 의미한다. 이는 데이터의 변화 패턴을
> 즉, 주성분 벡터를 이용하면 적은 축(특성 개수)으로도 원래 데이터를 잘 설명할 수 있다.  
> PCA는 데이터를 주성분 축으로 “투영(projection)”했다가,  
> 그 투영값을 다시 주성분 축의 선형결합으로 되돌려  
> 원래 공간에서의 근사값으로 복원한다.  

PCA에서, 각각의 주성분 벡터 z의 수식적 정의는 다음과 같다.  
이미지 하나 x(10000 차원 벡터)에 대해:  
> <img width="176" height="31" alt="image" src="https://github.com/user-attachments/assets/00ab9176-cda8-47d3-9ba9-1437b41a0754" />  
전개하면  
> <img width="522" height="188" alt="image" src="https://github.com/user-attachments/assets/0bf6ea55-71d3-4bf5-973a-e42406457d6d" />
* 평균을 빼는 과정(x−μ)의 이유는 “절대적인 밝기” 제거, 변화 패턴만 보겠다는 의미이다. PCA는 항상 평균 0 기준에서 동작한다.  
* 주성분 가중치 벡터 w는 데이터의 분산이 가장 커지는 방향(고유벡터)에 의해 결정된다.  
  wki는 ‘i번째 픽셀이 k번째 주성분 방향을 구성하는 데 얼마나 기여하는가’를 나타내는 값으로,  
  **전체 데이터셋**의 공분산에 영향을 받는다.  
* PCA의 1번 주성분은 다음을 만족한다.  ​
> <img width="481" height="185" alt="image" src="https://github.com/user-attachments/assets/1b75a06f-fa6b-4f68-bf7f-c3f7c7019be8" />

데이터

## PCA 클래스
> python의 클래스인 PCA는  
> 고차원 데이터를 분산이 최대가 되도록 회전시킨 뒤,  
> 그 좌표계에서 일부 축만 남겨 데이터를 표현·복원하는 변환기(transformer)이다.

**핵심 속성(attribute)**  
* *n_components*: 주성분 개수  
* random_state: Numpy 난수 시드값
    
* *components_*  
  > ```python
  > pca.components_.shape  
  > #(n_components, n_features)
  > ```
   
  각 행: 하나의 주성분 축  
  각 값: 픽셀(특성)의 가중치 wki  
  수학적 의미: 공분산 행렬의 고유벡터, 분산을 최대화하는 방향
  
* *mean_*  
  전체 데이터셋의 평균 벡터  
  transform시 항상 빼줌  
  의미: "절대 밝기 제거(이미지 기준)", 변화량(패턴)만 분석  

* *explaned_variance_*  
  각 주성분이 설명하는 분산의 크기  
  고유값에 해당  

* *explaned_variance_ratio_  
  전체 분산 중 각 주성분이 차지하는 비율
  차원 축소 판단 기준

* singular_values_  
  SVD(특이값 분해)에 나오는 특이값  
  고유값과 직접 연결됨  

**transform()**
>```python
>pca_data = pca.transform(data_set) # 2차원 격자 구조 입력 요구
>print(pca_data.shape)
># (n_samples, n_components)
>```
transform(n_components) 메서드는 데이터셋을 n개 주성분에 대해 transform한다.  
변환된 데이터 셋에서 각 샘플의 특성은 그 샘플의 각 주성분에 대한 좌표값(zk)이다.  
즉, 해당 주성분 좌표계에서 한 샘플의 위치이다.

**inverse_transform()**
아래 수식을 바탕으로 데이터를 원 형태로 복구한다.  
> <img width="476" height="170" alt="image" src="https://github.com/user-attachments/assets/ea89d609-f3c2-4b3e-9176-b99f7c9aeeaf" />
      
## 코드 전문
```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape)

import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다.
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols,
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
draw_fruits(pca.components_.reshape(-1, 100, 100))

print(fruits_2d.shape)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

## 원본 데이터 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")

## 설명된 분산
print(np.sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)
plt.show()

## 다른 알고리즘과 함께 활용
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
target = np.array([0] * 100 + [1] * 100 + [2] * 100)
from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

print(pca.n_components_)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)

print(np.unique(km.labels_, return_counts=True))
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```
