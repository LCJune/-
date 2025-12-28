## 핵심 키워드 
**드롭 아웃(Dropout)**    
은닉층에 있는 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막는 기법.  
이전 층의 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과대하게 의존하는 것을 줄일 수 있고,  
모든 입력에 대해 주의를 기울이게 된다. 일부 뉴런의 츌럭이 없을 수 있다는 것을 감안함으로써, 신경망은 더 안정적인 예측을 만든다.
드롭 아웃은 훈련 중에 적용되며, 평가나 예측에서는 적용하지 않는다. 케라스는 이를 자동으로 처리한다.  

**콜백(Callback)**    
케라스 모델 훈련 도중에 어떠한 작업을 수행할 수 있도록 도와주는 도구이다.  
대표적으로 최상의 모델을 자동으로 저장해주는 ModelCheckpoint('best-model.keras', save_best_only = True),  
검증 점수가 더 이상 향상되지 않으면 학습을 일찍 종료하는 EarlyStopping 등이 있다.

**조기 종료(Early Stopping)**   
검증 점수가 더 이상 감소하지 않고 상승하여 과대적합이 일어나면 훈련을 계속 진행하지 않고 멈추는 기법이다.  
계산 비용과 시간을 절약하는 데에 도움이 된다.

## 핵심 패키지와 함수
### KERAS
**Dropout**  
드롭아웃 층을 만드는 클래스이다.  
첫 번째 매개변수로 드롭아웃 할 비율(r)을 지정한다.  
드롭아웃 하지 않는 뉴런의 출력은 1/(1-r)만큼 증가시켜 출력의 총합이 같도록 만든다.

**save_weights()**    
모든 층의 가중치와 절편을 파일에 저장하는 메서드이다.  
첫 번째 매개변수에 저장할 파일을 지정한다.  
*save_format*: 저장할 파일 포맷을 지정한다. 기본적으로 HDF5 포맷으로 가중치와 절편을 저장한다. 파일 이름은 반드시 '.weights.5h'로 끝나야 한다.

**load_weights()**  
save_weights()로 저장된 모든 층의 가중치와 절편을 파일에 읽는 메서드이다.  
첫 번째 매개변수에 읽을 파일을 지정한다.

**save()**  
모델 구조와 모든 가중치와 절편을 파일에 저장한다.  
첫 번째 매개변수에 저장할 파일을 지정한다.
*save_format()** 저장할 파일 포맷을 지정한다. 기본적으로 케라스 3.x 포맷으로 저장하며, 파일 이름은 '.keras'로 끝나야 한다.  

**load_model()**   
mode.save()로 저장된 모델을 로드한다.  
첫 번째 매개변수에 읽을 파일을 지정한다.  

**ModelCheckpoint**  
Callback 클래스로, 케라스 모델과 가중치를 일정 간격으로 저장한다.  
첫 번째 매개변수에 저장할 파일을 지정한다.  
*monitor*: 모니터링 할 지표를 지정한다. 기본값은 'val_loss'로, 검증 손실을 관찰한다.  
*save_weights_only*: 기본값은 False로, 전체 모델을 저장한다. rue로 지정하면 모델의 가중치와 절편만 저장한다.  
*save_best_only*: True로 지정하면 가장 낮은 검증 점수를 만드는 모델을 저장한다.  
                  monitor의 기본값이 val_loss, 즉 검증 손실이기 때문에, 이 경우에는 검증 점수가 낮을수록 좋다.  
**EarlyStopping**  
Callback 클래스로, 
관심 지표가 더이상 향상하지 않으면 훈련을 중지한다.  
*monitor*: 모니터링할 지표를 지정한다. 기본값은 'val_loss'로 검증 손실을 관찰한다.   
*patience*: 모델이 더 이상 향상되지 않고 훈련을 지속할 수 있는 최대 에포크 횟수를 지정한다.  
*restore_best_weights: 최상의 모델 가중치를 복원할지 지정한다. 기본값은 False이다.  

### Numpy
**argmax()**  
배열에서 축을 따라 최댓값의 인덱스를 반환한다.  
axis: 최댓값을 찾을 축을 지정한다. 기본값은 None으로, 전체 배열에서 최댓값을 찾는다.  
      axis = n-1 (n번째 차원)과 같이 지정한다.  

```python
import keras

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input/255.0
from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42
)

def model_fn(a_layer = None):
  model = keras.Sequential()
  model.add(keras.layers.Input(shape = (28,28)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(100, activation = 'relu'))
  if a_layer:
    model.add(a_layer)
  model.add(keras.layers.Dense(10, activation = 'softmax'))
  return model

model = model_fn()
model.summary

model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
"""
fit() 메서드는 History 클래스 객체를 반환한다.
아래 histroy 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어 있다.
"""
history = model.fit(train_scaled, train_target, epochs = 5, verbose = 0)
# verbpse = 0으로 지정시 훈련 과정을 표시하지 않는다.
print(history.history.keys())

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0, validation_data = (val_scaled, val_target))
print(history.history.keys())
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend()
plt.show()

model = model_fn()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0,
                     validation_data = (val_scaled, val_target))
print(history.history.keys())
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.xlabel('loss')
plt.legend()
plt.show()

model = model_fn(keras.layers.Dropout(0.3))
model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_scaled, train_target, epochs = 11, verbose = 0,
                     validation_data = (val_scaled, val_target))
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.save('model_whole.keras')
model.save_weights('model.weights.h5')

!ls -al model* # !를 쓰면 셀 명령을 실행

model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model.weights.h5')

import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis = -1)
print(np.mean(val_labels == val_target))

model = keras.models.load_model('model_whole.keras')
model.evaluate(val_scaled, val_target)

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2,
                                                  restore_best_weights = True)
history = model.fit(train_scaled, train_target, epochs = 20, verbose = 0,
                    validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt
print(early_stopping_cb.stopped_epoch)
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.evaluate(val_scaled, val_target)

```
