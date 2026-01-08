## LSTM의 구성 요소
> <img width="552" height="285" alt="image" src="https://github.com/user-attachments/assets/e959dc77-91c8-4002-9421-ce970790291a" />


LSTM은 은닉 상태(Hidden State, 이하 ht)외에 셀 상태(Cell State, 이하 ct)를 추가로 가진다.  
* **셀 상태(Cell State)**   
  * 타임스텝을 따라 거의 선형적으로 흐르며 정보 보존   
  * 필요 없는 정보는 제거하고, 중요한 정보만 유지하도록 게이트가 제어  

* **은닉 상태(Hidden State)**
  * 외부로 출력되는 상태   
  * 현재 시점의 요약 정보   
  * 다음 LSTM 셀 및 출력층으로 전달됨  
  
LSTM 셀은 특정 기능을 가진 뉴런 집합들로 구성된다. 뉴런 집합은 역할에 따라 4종류로 나뉜다.  
1. Forget Gate: 이전 기억을 얼마나 유지할지 조절함  
2. Input Gate: 새 정보를 반영할 비율을 조절함  
3. Output Gate: 외부로 노출할 정보를 조절함
4. Candidate: 현재 시점에 새로 추가될 수 있는 기억의 내용을 생성함  

* Cadndiate가 새로운 내용을 제안    
  -> 입력 xt와  ht-1를 바탕으로 현재 시점에서 유용하다고 판단되는 정보 벡터 생성    
* Input Gate와 곱해져 선택적으로 Cell State에 반영    
* Forget Gate가 Cell State의 각 성분에 대해 유지 비율(0~1)을 결정하는 제어 신호를 생성  
  -> 현재 입력 정보에 대해 이전 기억중 어떤 부분은 이제 의미가 없는지를 학습  

gate, 혹은 candidate라는 이름으로 부르고 있으나, 이들은 모두 뉴런의 집합이다.  
이 객체들을 구성하는 개별 요소는 weight와 bias로 입력에 대한 선형 변환을 수행하고 활성화 함수를 적용하는 뉴런이다.  

## LSTM의 기능 단위  
> LSTM의 기능 단위는 3개의 gate와 candidate이며, hidden 차원마다 하나씩 존재한다.  
> 즉, unit = 128이면, 각 기능 단위들 또한 128개씩 존재한다.  
> LSTM의 units는 모델이 시퀀스를 표현하기 위해 사용하는 잠재 상태 공간의 차원으로,  
> 정보 해석 관점에서는 동시에 추적할 수 있는 잠재 특성의 수로 비유할 수 있다.  
> hidden state 차원마다 시퀀스를 표현하는 축이 다르기 때문에, 모든 gate와 candidate는 서로 다른 가중치와 절편을 가진다.  
> LSTM이 주어진 시계열 데이터를 다차원적으로 해석하고 관리할 수 있게 만드는 원리인 것이다.

### 망각 게이트(forget gate)
> <img width="540" height="227" alt="image" src="https://github.com/user-attachments/assets/17aa59d2-60b5-415c-adb2-2e13efc1cd1b" />



