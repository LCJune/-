## LSTM의 구조  
> <img width="1826" height="564" alt="image" src="https://github.com/user-attachments/assets/13849a11-3e7a-4812-b9f6-5e6978d37102" />  


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

Cadndiate가 새로운 내용을 제안  
-> 입력 xt와  ht-1를 바탕으로 현재 시점에서 유용하다고 판단되는 정보 벡터 생성  
Input Gate와 곱해져 선택적으로 Cell State에 반영   
Forget Gate가 Cell State에서 
* **망각 게이트(forget gate)**
  
