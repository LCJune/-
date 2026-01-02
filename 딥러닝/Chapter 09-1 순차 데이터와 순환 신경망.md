## 핵심 키워드
**순차 데이터(Sequential Data)**: 텍스트나 시계열 데이터(Time Series Data)와 같이 순서에 의미가 있는 데이터.  

**피드포워드 신경망(Feedforward Neural Network)**: 입력 데이터의 흐름이 앞으로만 전달되는 신경망.  
완전 연결 신경망, 합성곱 신경망이 모두 피드포워드 신경망에 속한다.  

**순환 신경망(Recurrent Neural Network)**: 다음 샘플을 위해 이전 데이터가 신경망 층에 순환되는 신경망.  
뉴런의 출력이 다시 자기 자신으로 전달된다.  

**타임 스텝(Time Step)**: 순환 신경망에서 샘플을 처리하는 한 단계    

## 순환 신경망(Recurrnt Neural Network, RNN)
> <img width="2560" height="854" alt="image" src="https://github.com/user-attachments/assets/6c731636-0c64-4fdb-b43f-d6243949d4e5" />  

다음 샘플을 위해 이전 데이터가 신경망 층에 순환되는 신경망. 뉴런의 출력이 다시 자기 자신으로 전달된다.  
순차 데이터를 다루기 위해서는 샘플 간의 순서가 중요하다. 즉, 이전에 처리했던 샘플의 정보를 다음 샘플을 처리할 때 다시 사용해야 한다.  

> <img width="472" height="230" alt="image" src="https://github.com/user-attachments/assets/37718e93-ec47-45bf-aea1-ba5002ee5ac8" />  

* 타임 스텝(Time Step): 샘플을 처리하는 한 단계  
* 셀(cell): RNN에서 하나의 층을 표현하는 객체  
* 은닉 상태(hidden state): 셀의 출력  

RNN 또한 다른 신경망들과 기본 구조는 같다. 입력에 가중치를 곱하고 활성화 함수를 통과시켜 다음 층으로 보낸다.  
차이점은 층의 출력(은닉 상태)을 다음 타임 스텝에 이용한다는 것이다.  

RNN의 은닉층 활성화의 함수에는 하이퍼볼릭 탄젠트(tanh) 함수가 주로 쓰인다.
> <img width="646" height="430" alt="image" src="https://github.com/user-attachments/assets/58e280d9-3974-4f2c-a4f1-e29a47820e4e" />

RNN은 출력에 곱해지는 가중치 외에, 이전 타임 스텝에 곱해지는 가중치를 추가로 가진다.  
> <img width="860" height="334" alt="image" src="https://github.com/user-attachments/assets/8f67e887-f4b6-4d7d-8791-59cc4b77c509" />   

이때, 모든 타임 스텝에서 은닉 상태에 곱해지는 가중치 wh는 하나다. h0의 경우는 0으로 초기화 한다.  

> <img width="220" height="342" alt="image" src="https://github.com/user-attachments/assets/78babbe6-06ae-4487-bc51-eaf69b94e9b8" />

이전 타임 스텝의 은닉 상태는 다음 타임 스텝의 뉴런에 완전히 연결된다.  
즉, 하나의 은닉 상태를 출력하면 해당 출력을 같은 셀의 모든 뉴런에 전달한다.  
따라서 셀 전체의 wh의 개수는 (셀의 뉴런 수)^2가 된다.  
> 모델 파라미터 수 = wx + wb + 절편

RNN에서는 하나의 샘플을 시퀀스(Sequence)라고 부른다.  
시퀀스는 일반적으로 2개의 차원을 가진다. 시퀀스 길이(Time Step)과 '단어표현'이라는 축을 가진다.  
하나의 샘플은 순환층을 통과하면 1차원 배열로 바뀐다.  

순환층은 기본적으로 마지막 타임스텝의 은닉 상태만 출력으로 내보낸다.   
그러나 순환층이 여러개일 경우, 다음 순환층은 이전 순환층의 모든 타임 스텝에서의 은닉 상태를 필요로 하므로,  
모든 은닉 상태를 다음 순환층으로 전달한다.  
