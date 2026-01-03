## 핵심 키워드
**말뭉치(Corpus)**: 자연어 처리에서 사용하는 텍스트 데이터의 모음, 즉 훈련 데이터셋.  

**토큰(Token)**: 텍스트에서 공백으로 구분되는 단어 혹은 단어의 일부분.  
종종 소문자로 변환하고 구둣점은 삭제한다.  

**원-핫 인코딩(One-Hot Incoding)**: 어떤 클래스에 해당하는 원소만 1이고 나머지는 모두 0인 벡터.  
정수로 변환된 토큰을 원-핫 인코딩으로 변환하려면 어휘사전 크기의 벡터가 만들어진다. -> 메모리의 낭비가 심해진다.  

**단어 임베딩(Word Embedding)**: 정수로 변환된 토큰을 비교적 작은 크기의 실수 밀집 벡터로 변환한다.  
이런 밀집 벡터는 단어 사이의 관계를 표현할 수 있기 때문에 자연어 처리(NLP)에서 좋은 성능을 발휘한다.    

## RNN의 훈련 과정
RNN의 훈련 과정은 다음과 같다.    
1. 정수 토큰 시퀀스를 입력으로 받아 임베딩 벡터 시퀀스로 변환(랜덤 초기화)  
2. RNN이 벡터 시퀀스를 이용해 결과를 출력(긍정문 or 부정문)  
3. 해당 출력값에 대한 loss 계산 후, backpropagation 실행  
4. 임베딩 벡터 이동  

위 훈련 과정을 통해 비슷한 단어들은 임베딩 벡터 안에서 가까워진다.  
hidden state를 비슷한 방향으로 이동시키므로, loss 관점에서 비슷한 업데이트를 받기 때문이다.  

임베딩 벡터의 의미는 단어가 문맥 속에서 출력에 기여하는 방식으로 정의되며,  
비슷한 역할을 하는 단어들은 학습 과정에서 유사한 gradient를 받아 벡터 공간에서 자연스럽게 가까워진다.  

## 핵심 패키지와 함수
### KERAS
* **pad_sequences()**  
  시퀀스 길이를 맞추기 위해 패딩을 추가한다. 이 함수는 (샘플 개수, 타임스텝 개수) 크기의 2차원 배열을 기대한다.

* **to_categorical()**
  정수 시퀀스를 원-핫 인코딩으로 변환한다. 토큰을 원-핫 인코딩하거나 타깃값을 원-핫 인코딩 할 때 사용한다.  
  *num_classes*: 클래스 개수를 지정한다. 지정하지 않으면 데이터에서 자동으로 찾는다.

* **SimpleRNN
  케라스의 기본 순환층 클래스이다.
  첫 번째 매개변수에 뉴런의 개수를 지정한다.  
  *activation*: 활성화 함수를 지정한다. 기본값은 하이퍼볼릭 탄젠트인 'tanh'이다.
  *dropout*: 입력에 대한 드롭아웃 비율을 지정한다.
  *return_sequences*: 모든 타임스텝의 은닉 상태를 출력할지 결정한다. 기본값은 False이다.

* **Embedding**
  단어 임베딩을 위한 클래스이다.
  첫 번째 매개변수에서 어휘 사전의 크기를 지정한다.
  두 번째 매개변수에서 Embedding 층이 출력할 밀집 벡터의 크기를 지정한다.

## 코드 전문
```python
from keras.datasets import imdb
# 영화 리뷰 댓글 db

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=200)
# 어휘 사전 단어 200개, 단어를 등장 횟수 순서대로 나열한 다음, 가장 많이 나온 200개 단어를 어휘 사전에 포함

print(train_input.shape, test_input.shape)
# (25000,) (25000, )

print(len(train_input[0]))
# 218

print(len(train_input[1]))
# 189

print(train_input[0])
# 첫 번째 댓글의 모든 token 출력

print(train_target[:20])
'''
[1 0 0 1 0 0 1 0 1 0 0 0 0 0 1 1 0 1]
1: 긍정, 0: 부정
'''

from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

import numpy as np
length = np.array([len(x) for x in train_input])

print(np.mean(length), np.median(length))
# 239.00925 178.0
# 평균이 중간값보다 높다 -> 오른쪽 끝에 아주 큰 데이터가 있다.

import matplotlib.pyplot as plt
plt.hist(length)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

# 리뷰 데이터의 길이를 맞추는 pad_sequences() 함수
from keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen = 100)
'''
토큰의 수가 100 미만이면 0(패딩)으로 채워서 100으로 맞춤
토큰의 수가 100 초과면 앞 부분의 토큰을 자름
뒷부분을 자르고 싶다면 truncating = 'pre'가 아닌 'post'로 지정
'''

print(train_seq.shape)
# (20000, 100)

val_seq = pad_sequences(val_input, maxlen = 100)

import keras
model = keras.Sequential()
model.add(keras.layers.Input(shape=(100, 200)))
"""
100: 토큰 수, 200: 어휘사전 단어 수
토큰에 매칭된 정수값이 클수록 활성화 함수에서 더 큰 값을 만들어낸다.
따라서 정숫값의 크기 속성을 없애기 위해 원-핫 인코딩 방식을 사용해야 한다.
각 토큰은 자신의 200개의 칸 중에서 자신의 토큰값에 해당하는 인덱스만 1로,
나머지는 0으로 채워진 배열을 갖는다.
"""
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# 정수 시퀀스를 원-핫 인코딩 방식으로 변환
train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape)
# (20000, 100, 200)

print(np.sum(train_oh[0][0][:12]))
# [0. 0. 0. 0. 0. 0 .0 .0 .0 .0. 1. 0.]
# 11번째 원소만 1이다. -> 첫 번째 샘플의 첫 번째 토큰의 매칭 정수값은 11

print(np.sum(train_oh[0][0]))
# 1.0 -> 원-핫 인코딩 완료

val_oh = keras.utils.to_categorical(val_seq)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras',
                                                save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

## 단어 임베딩 방식 RNN
(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

model_emb = keras.Sequential()
model_emb.add(keras.layers.Input(shape = (100, )))
model_emb.add(keras.layers.Embedding(500, 16)) # (토큰 수, 임베딩 벡터 크기)
model_emb.add(keras.layers.SimpleRNN(8))
model_emb.add(keras.layers.Dense(1, activation='sigmoid'))

model_emb.summary()

model_emb.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras',
                                                save_best_only= True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3,
                                                  restore_best_weights=True)

history = model_emb.fit(train_seq, train_target, epochs = 100, batch_size = 64,
                        validation_data = (val_seq, val_target),
                        callbacks = [checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

```
