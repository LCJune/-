```python
import matplotlib.pyplot as plt #matplotlib의 pyplot 함수를 plt로 줄여서 사용
from sklearn.neighbors import KNeighborsClassifier #사이킷런 라이브러리에서 K-최근접 이웃 알고리즘을 구현한 클래스인 KNeighborsClassifier 임포트

"""### 도미 데이터 준비하기"""
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

"""### 빙어 데이터 준비하기"""
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l,w] for l, w in zip(length, weight)] #zip 함수를 이용해 length, weight의 값을 하나씩 가져와 fish_data에 한 쌍씩 저장
fish_target = [1] * 35 + [0] * 14 #정답 배열

kn = KNeighborsClassifier() #KNeighborsClassifier 객체 생성

kn.fit(fish_data, fish_target) #fit 메소드를 이용해 객체 훈련

kn.score(fish_data, fish_target) #모델의 분류 정확도를 평가하는 score 메소드

plt.scatter(bream_length, bream_weight) #그래프에 산점도를 표시하는 함수. 순서대로 X축, Y축.
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^') 
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
<img width="580" height="432" alt="Image" src="https://github.com/user-attachments/assets/fac5b806-c1b0-4f34-9207-1213d3bbc5a9" />

```python
kn.predict([[30,600]]) '''학습된 기준을 이용해 새로운 데이터를 분류하는 함수. 2차원 배열(리스트)을 매개변수로 받는다.
                          이 구문의 출력값은 1, 즉 도미로 분류되었다는 의미이다. 이를 통해 모델이 학습된 기준을 토대로 데이터를 분류하였음을 알 수 있다. 
                       '''
```

