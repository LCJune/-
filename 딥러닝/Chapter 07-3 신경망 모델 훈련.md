## 핵심 키워드 

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
