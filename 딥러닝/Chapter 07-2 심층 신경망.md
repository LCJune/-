<img width="500" height="153" alt="image" src="https://github.com/user-attachments/assets/ac52463a-90aa-431e-9d5e-22e7bd076e31" />## 핵심 키워드
**심층 신경망(Deep Neural Network, DNN)**  
> 2개 이상의 층을 포함한 신경망을 이르는 말이다. 종종 다층 인공 신경망, 심층 신경망, 딥러닝을 같은 의미로 사용한다.  

**렐루 함수(ReLU Function)**  
> 입력 값이 양수일 경우 입력을 그대로 통과시키고, 0 이하일 경우 0을 출력하는 함수. 즉, f(x) = max(0, x)
> sigmoid 함수는 층이 많을수록 활성화 함수의 양 끝에서 변화가 작기 때문에 학습이 어려워지지만, 렐루함수는 그런 문제가 없고 계산이 간단하다.  


## 옵티마이저(Optimizer)
머신러닝·딥러닝 모델에서 손실 함수(loss function)를 최소화하도록 모델의 파라미터(가중치, 편향)를 어떻게, 어떤 규칙으로 업데이트할지 정의한 알고리즘을 의미한다.  
> <img width="472" height="376" alt="image" src="https://github.com/user-attachments/assets/1f265883-e1b0-4e3f-a03b-37e8e4b633ac" />
 
대표적인 옵티마이저의 종류로는 SGD, Momentum, Nesterove Momentum, Adagrad, RMSprop, Adam 등이 있다.

위 옵티마이저들은 확률적 경사 하강법을 기본 구조로 가지고 있다. 가장 일반적인 형태로는 다음과 같다.  
<img width="494" height="157" alt="image" src="https://github.com/user-attachments/assets/6b1dbf8c-0e56-4158-a1a3-19f8e70d8455" />

### 1. SGD
> <img width="470" height="117" alt="image" src="https://github.com/user-attachments/assets/d45513ad-1730-41e6-85d4-3cf6b7416f31" />
> <img width="268" height="24" alt="image" src="https://github.com/user-attachments/assets/14c762f7-acd1-4312-a1a7-8b83aa95c62e" />


### 2. Momentum
물리학의 '관성'의 원리를 모델 학습에 적용한 방식으로, 이전의 그레디언트를 가속도처럼 활용한다.  
> <img width="500" height="153" alt="image" src="https://github.com/user-attachments/assets/66a8c03c-3ce5-4ee5-b08d-acdaa7f713d9" />

### 3. Nesterov Momentum
일반적인 Momentum의 경우, 
### 4. Adagrad
> <img width="526" height="145" alt="image" src="https://github.com/user-attachments/assets/3dece0cb-1e92-4738-b933-2a032833b214" />

### 5. RMSprop
>

### 6. Adam
> <img width="512" height="139" alt="image" src="https://github.com/user-attachments/assets/03bf518a-2a28-463b-98ea-6947de3c3b83" />
> <img width="500" height="138" alt="image" src="https://github.com/user-attachments/assets/a7fed19b-8b25-4c31-b6b6-092ecf885b13" />








## 핵심 패키지와 함수
### KERAS
**add()**  
케라스 모델에 층을 추가하는 메서드이다. keras.layers 패키지 아래에 있는 층의 객체를 입력받아 신경망 모델에 추가한다.  
add() 메서드를 호출하여 전달한 순서대로 층이 늘어난다.  

**summary()**  
케라스 모델의 정보를 출력하는 메서드이다. 모델에 추가된 층의 종류와 순서, 모델 파라미터 개수를 출력한다.  
층을 만들 때 name 매개변수로 이름을 지정하면 summary() 메서드 출력에서 구분하기 쉽다.  

**Flatten()**
층 객체 클래스 중 하나로, 배치 차원을 제외한 나머지 입력 차원을 모두 일렬로 펼친다.  
입력에 곱해지는 가중치나 절편이 없어 모델의 성능에 기여하지는 않지만,  
층처럼 입력층과 은닉층 사이에 추가하기 떄문에 층이라 부른다. summry()에도 출력된다.  

**SGD**  
가장 기본적인 확률적 경사 하강법 옵티마이저 클래스이다. 기본적으로 미니배치를 사용한다.
*learning_rate*: 학습률을 지정하며 기본값은 0.01이다.  
*momentum*: 0 이상의 값을 지정하면 모멘텀 최적화를 수행한다.  
*nestrov**: 매개변수를 True로 설정하면 네스테로프 모멘텀 최적화를 수행한다.  

**Adagrad**  
Adagrad 옵티마이저 클래스이다.  
*learning_rate*: 학습률을 지정하며 기본값은 0.001이다.  
*initial_accumulator_value*: Adagrad는 그레디언트 제곱을 누적하여 학습률을 나누는데, 그 누적 초기값을 지정하는 매개변수이다. 기본값은 0.1이다.  

**RMSprop**  
RMSprop 옵티마이저 클래스이다.  
*learning_rate*: 학습률을 지정하며 기본값은 0.001이다.
*rho*: RMSprop 역시 Adagrad처럼 그레디언트 제곱으로 학습률을 나누지만, 최근의 그레디언트를 사용하기 위해 지수 감소를 사용한다. rho는 그 감소 비율을 지정하며 기본값은 0.9이다.  

**Adam**
Adam 옵티마이저 클래스이다. 
*learning_rate*: 학습률을 지정하며 기본값은 0.001이다.  
*beta_1*: 모멘텀 최적화에 있는 그레디언트의 지수 감소 평균을 조절한다. 기본값은 0.9이다.  
*beta_2*: RMSprop에 있는 그레디언트 제곱의 지수 감소 평균을 조절한다. 기본값은 0.999이다.  

## 코드 전문
```python
import keras
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42
)

print(train_scaled.shape, train_target.shape)

inputs = keras.layers.Input(shape=(784,))

### 각 층 객체별 instance를 만들어 전달하는 방법
dense1 = keras.layers.Dense(100, activation = 'sigmoid') # 밀집 층
dense2 = keras.layers.Dense(10, activation = 'softmax') # 출력 층
model = keras.Sequential([inputs, dense1, dense2]) # 입력 층을 맨 앞에 두고, 출력층을 가장 마지막에 두어야 함

### 생성자에 한 번에 전달하는 방법
model = keras.Sequential([
    keras.layers.Input(shape = (784, )),
    keras.layers.Dense(100, activation = 'sigmoid', name = 'hidden_layer'),
    keras.layers.Dense(10, activation = 'softmax', name = 'output_layer')
], name = '패션 MNIST 모델')

### add 메서드를 사용하는 방법
model = keras.Sequential()
model.add(keras.layers.Input(shape = (784, )))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'sigmoid', name = 'hidden_layer'))
model.add(keras.layers.Dense(10, activation = 'softmax', name = 'output_layer'))


model = keras.Sequential()
model.add(keras.layers.Input(shape = (28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
train_scaled  = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42
)

model = keras.Sequential()
model.add(keras.layers.Input(shape = (28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_scaled, train_target, epochs = 5)

model.evaluate(val_scaled, val_target)
```
