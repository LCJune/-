## K-means clustering
: K-means clustering 알고리즘은 최적의 클러스터를 구성하는 알고리즘이다. 작동 방식은 다음과 같다.
1. 랜덤으로 클러스터 중심을 선정한다.
2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터에 할당한다.
3. 클러스터 내 샘플들의 평균값으로 클러스터 중심(좌표)를 이동한다.
4. 클러스터 중심에 변화가 없을 때까지(수렴할 때까지) 2번으로 돌아가 반복한다.

## 핵심 키워드
**클러스터 중심(cluster center, centroid)  
> K-평균 알고리즘이 만든 클러스터 내 샘플들의 특성 평균값

**이너셔(inertia)**
> 클러스터 내 샘플에서 클러스터 중심까지의 거리의 제곱 합.   
> K-mean 알고리즘에서 클러스터 중심이 수렴할수록 inertia의 값은 줄어든다.  
> 클러스터 개수에 따라 inertia의 감소가 꺾이는 지점이 생기게 되는데,
> 해당 지점이 적절한 클러스터 개수가 될 수 있다.  
> 위 방식으로 최적의 클러스터를 찾는 방법을 엘보우 방법이라고도 한다.
  
## 핵심 패키지와 함수
### scikit-learn
> **KMeans**: k-평균 알고리즘 클래스이다.  
>> *n_clusters*: 클러스터 개수를 지정한다. 기본값은 8이다.
>
>> *init*: centroid의 초기화 방식을 지정한다.  
>> 기본값인 'k-means++'인 경우 서로 멀리 떨어진 점들로 초기화한다. (수렴 안정성 증가)  
>> 'random'인 경우, 데이터 중 무작위로 k개를 선택한다.
>
>> *n_init*: K-means 알고리즘의 전체 반복 횟수를 지정한다. 기본값은 'auto'이다.  
>> *n_init*이 'auto'일 때, *init* 값이 'k-means++'라면 1, 'random'이라면 10으로 지정된다.  
>> 매 시행마다 랜덤한 centroid 초기값을 가지고 알고리즘을 *n_init*만큼 반복하고,  
>> inertia가 가장 작은 모델을 선택한다.  
## 코드 전문
```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_2d)

# print(km.labels_)
# print(np.unique(km.labels_, return_counts = True))

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

# draw_fruits(fruits[km.labels_ == 0])
# draw_fruits(fruits[km.labels_ == 1])
# draw_fruits(fruits[km.labels_ == 2])

"""
KMeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_ 속성에
저장된다. 이미지 기반 KMeans에서 클러스터 중심은,
클러스터에 속한 모든 이미지의 각 픽셀 위치별 평균값으로 구성된 벡터이며,
이를 reshape 하면 “평균 이미지”로 시각화할 수 있다.
reshape가 필요한 이유는 imshow() 메서드가 2차원 격자구조를 입력으로 요구하기 때문이다.
즉 입력되는 데이터는 (height, width) 혹은 (row, column)의 2차원 배열 형태여야 한다.
"""

# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 3)

"""
transform() 메서드는 훈련 데이터 샘플에서 클러스터 중심까지의 거리를 반환한다.
fit() 메서드와 마찬가지로 2차원 배열 형태의 입력을 요구한다.
따라서 아래 입력을 fruits_2d[100] (10000,)의 형태가 아닌
fruits_2d[100:101] (1,10000)의 형태로 주었다.
이와 같은 차이가 발생하는 이유는 Numpy의 인덱싱 규칙 때문이다.
정수 인덱싱(fruits_2d[100])은 100번째 행이라는 하나의 1차원 벡터를 가져온다.
반면 슬라이싱(fruits_2d[100:101]은 100번째 행부터 101번째 행 전까지 배열을 자르는 것으로,
차원이 보존된다. 즉 행이 1개인 행렬을 가져온다.
"""
print(km.transform(fruits_2d[100:101]))

## predict() 또한 2차원 배열을 입력으로 요구한다.
print(km.predict(fruits_2d[100:101]))
# output: [0](pineapple)
draw_fruits(fruits[100:101])

inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters= k, n_init='auto', random_state = 42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```
