## 핵심 패키지와 함수  
### KERAS
* **Conv2D**  
  입력의 너비와 높이 방향의 합성곱 연산을 위한 클래스이다.  
  첫 번째 매개변수는 합성곱 필터의 개수이다.  
  *kernel_size*: 필터의 커널 크기를 지정한다. 가로세로 크기가 같은 경우 정수 하나로,  
  다른 경우 (높이, 너비)로 구성된 정수 튜플을 지정할 수 있다. 일반적으로 커널의 가로세로 크기는 동일하다.  
  커널의 깊이는 입력의 깊이와 동일하므로 지정하지 않는다.  

  *strides*: 필터의 이동 간격을 지정한다. 가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정한다.  
  일반적으로 가로세로 스트라이드 크기는 같다. 기본값은 1이다.  

  *padding*: 입력의 패딩 타입을 지정한다. 기본값 'valid'는 패딩을 하지 않는다.  
  'same'은 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가한다.  

  *activation*: 합성곱 층에 적용할 활성화 함수를 지정한다.  

* **MaxPooling2D**     
  입력의 너비와 높이를 줄이는 풀링 연산을 구현한 클래스이다.  
  첫 번째 매개변수는 풀리의 크기를 지정하며, 가로세로 크기가 같은 경우 정수 하나로,  
  다른 경우 (높이, 너비)로 구성된 정수 튜플을 지정할 수 있다. 일반적으로 가로세로 크기는 같다.  
  *strides*: 풀링의 이동 간격을 지정한다. 기본값은 풀링의 크기와 동일하다. 즉, 입력 위를 겹쳐서 풀링하지 않는다.  
  *padding*: 입력의 패딩 타입을 지정한다. 기본값 'valid'로 지정하면 패딩을 하지 않는다.  
  'same'으로 지정 시 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가한다.  
  풀링에서 패딩을 지원하기는 하나, 일반적으로는 거의 쓰이지 않는다.  

* **plot_model()**    
  케라스 모델 구조를 주피터 노트북에 그리거나 파일로 저장한다.  
  첫 번째 매개변수에 케라스 모델 객체를 전달한다.  
  *to_file*: 파일 이름을 지정하면 그림을 파일로 지정한다.     
  *show_shapes*: True로 지정 시 층의 입력, 출력 크기를 표시한다. 기본값은 False이다.  
  *show_layer_names*: 매개변수를 True로 지정하면 층 이름을 출력한다. 기본값은 True이다.  

### matplotlib
* **bar()**    
  막대 그래프를 출력한다.  
  첫 번째 매개변수에 x축의 값을 리스트나 numpy 배열로 전달한다.  
  두 번째 매개변수에 y축의 값을 리스트나 numpy 배열로 전달한다.  
  *width*: 막대의 두께를 지정한다. 기본값은 0.8이다.  

## 코드 전문
```python
import keras

from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size = 0.2, random_state = 42)

model = keras.Sequential()
model.add(keras.layers.Input(shape = (28, 28, 1)))
model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu',
                              padding = 'same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size = 3, activation = 'relu',
                              padding = 'same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()
# keras.utils.plot_model(model, show_shapes = True)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',
                                                save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2,
                                        restore_best_weights = True)
history = model.fit(train_scaled, train_target, epochs = 20,
                    validation_data = (val_scaled, val_target),
                    callbacks = [checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28,28), cmap = 'gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠',
           '스니커즈', '가방','앵클 부츠']

import numpy as np

print(classes[np.argmax(preds)])
# 가방
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

model.evaluate(test_scaled, test_target)
```
