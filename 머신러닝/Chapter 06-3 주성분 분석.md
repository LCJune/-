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


## PCA 클래스
> python의 클래스인 PCA는  
> 고차원 데이터를 분산이 최대가 되도록 회전시킨 뒤,  
> 그 좌표계에서 일부 축만 남겨 데이터를 표현·복원하는 변환기(transformer)이다.

**핵심 속성(attribute)**  
* *components_*
  > pca.components_.shape
  > #(n_components, n_features)
