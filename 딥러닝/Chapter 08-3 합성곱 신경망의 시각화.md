## 핵심 키워드
**가중치 시각화**: 합성곱 층의 가중치를 이미지로 출력하는 것.  
CNN은 주로 이미지를 다루기 때문에 가중치가 시각적인 패턴을 학습하는지 알아볼 수 있다.  

**특성 맵 시각화**: 합성곱 층의 활성화 출력을 이미지로 그리는 것.  
가중치 시각화와 함께 비교하여 각 필터가 이미지의 어느 부분을 활성화 시키는지 확인할 수 있다.   

**함수형 API**: Keras에서 신경망 모델을 만드는 방법 중 하나.  
Model 클래스에 모델의 입력과 출력을 지정한다. 전형적으로 입력은 Input() 함수를 사용하여 정의하고,  
출력은 마지막 층의 출력으로 정의한다.  

## 핵심 패키지와 함수
### Keras
* **Model**  
  케라스 모델을 만드는 클래스이다.
  inputs: 첫 번째 매개변수로, 모델의 입력 또는 입력의 리스트를 지정한다.
  outputs: 두 번째 매개변수로, 모델의 출력 또는 출력의 리스트를 지정한다.
  *name*: 모델의 이름을 지정한다.

## 코드 전문
```python
# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만든다.
import keras
import tensorflow as tf

keras.utils.set_random_seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

# 이전 절에서 학습시킨 모델 다운로드
!wget https://github.com/rickiepark/hg-mldl2/raw/main/best-cnn-model.keras

import keras
model = keras.models.load_model('best-cnn-model.keras')
model.layers
'''
[<Conv2D name=conv2d, built=True>,
 <MaxPooling2D name=max_pooling2d, built=True>,
 <Conv2D name=conv2d_1, built=True>,
 <MaxPooling2D name=max_pooling2d_1, built=True>,
 <Flatten name=flatten, built=True>,
 <Dense name=dense, built=True>,
 <Dropout name=dropout, built=True>,
 <Dense name=dense_1, built=True>]
'''

conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)
# (3, 3, 1, 32) (32,)
# keras.Conv2D에서의 weights 구조 (kernel_height, kernel_width, in_channels, out_channels)
# PyTorch.Conv2D에서의 weights 구조 (out_channels, in_channels, kernel_height, kernel_width)

conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())
# -0.017617775 0.22690606

import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
```
> <img width="562" height="432" alt="image" src="https://github.com/user-attachments/assets/9ca8292b-1835-4ea3-b53b-07e076712f40" />  
> 가중치가 0을 중심으로 종 모양의 분포를 띠고 있다.

```python
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()
```
> <img width="1182" height="165" alt="image" src="https://github.com/user-attachments/assets/c3b793ce-2a8e-4812-82f3-8a13bc0cf479" />
> 가중치 값이 무작위로 나열된 것이 아니고, 어떠한 패턴을 가지고 있다.  
> 예를 들어, 두 번째 줄의 왼쪽에서 여덟 번째 가중치는 왼쪽 3픽셀의 값이 다른 픽셀보다 상대적으로 낮다.(어두운 부분일수록 값이 낮음)    
> 이 가중치는 오른쪽에 놓인 직선을 만나면 크게 활성화될 것이다.
```python
#비훈련 모델과의 비교
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Input(shape=(28,28,1)))
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                          padding='same'))
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)
# (3, 3, 1, 32)
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())
# 0.0053191613 0.08463709

plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
```
> <img width="568" height="432" alt="image" src="https://github.com/user-attachments/assets/6cfd09a8-cc6b-4021-848f-239ba6ca552a" />
> 훈련된 모델과 달리 가중치가 -0.15 ~ 0.15에서 비교적 고른 분포를 띠고 있다.  
> 케라스가 신경망의 가중치를 처음 초기화 할 때 균등 분포에서 랜덤하게 값을 선택하기 때문이다.

```python
# 함수형 API
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax')

hidden = dense1(inputs)
outputs = dense2(hidden)

func_model = keras.Model(inputs, outputs)
print(model.inputs)
'''
[<KerasTensor shape=(None, 28, 28, 1), dtype=float32, sparse=False, ragged=False, name=input_layer>]
conv_acti = keras.Model(model.inputs[0], model.layers[0].output)
'''

# 특성 맵 시각화
(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

plt.imshow(train_input[0], cmap='gray_r')
plt.show()
```
> <img width="416" height="413" alt="image" src="https://github.com/user-attachments/assets/ce0babf6-1e8d-4870-b674-879b2019e797" />

```python
ankle_boot = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
feature_maps = conv_acti.predict(ankle_boot) # 첫 번째 합성곱 층의 출력(특성 맵)

print(feature_maps.shape)
# (1, 28, 28, 32) 순서대로 (batch_size, out_height, out_width, filters)
# same 패딩을 사용하므로 입력과 출력의 (h, w) 크기가 같다.

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()
```
> <img width="1182" height="625" alt="image" src="https://github.com/user-attachments/assets/0e2fa502-37be-486e-84af-2cf272e2ce81" />  
> 32개의 필터로 인해 입력 이미지에서 강하게 활성화된 부분을 보여준다. 훈련된 모델의 가중치를 출력한 것과 대조해보자.      
> (0,6)의 특성맵은 전체적으로 밝은 색이므로, 전면이 모두 칠해진 영역을 감지한다.    
> 24번째 필터는 수직선을 감지하는 것처럼 보인다. 따라서 (2,7)의 특성 맵은 이 필터가 감지한 수직선이 강하게 활성화 되었다.  

```python
# 두 번째 합성곱 층으로, 한 번의 풀링이 이루어진 특성 맵에 대해 합성곱 연산을 한다.
conv2_acti = keras.Model(model.inputs, model.layers[2].output)
feature_maps = conv2_acti.predict(ankle_boot)

print(feature_maps.shape)
# (1, 14, 14, 64) < 풀링이 이루어져 가로세로 크기가 절반으로 줄었다. 두 번째 합성곱 층의 필터 개수는 64개이다.

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()
```
> <img width="949" height="944" alt="image" src="https://github.com/user-attachments/assets/dc87365c-069a-4801-bd80-610710c8e9b6" />  
> 두 번째 합성곱 층의 필터 크기는 (3, 3, 32)이다.(필터의 깊이는 입력의 길이와 같다.)   
> 두 번째 합성곱 층의 첫 필터가 앞서 출력한 32개의 특성 맵과 곱해져 첫 번째 특성 맵이 된다.  
> 이렇게 계산된 출력은 (14, 14, 32) 특성 맵에서 어떤 부위를 감지하는지 직관적으로 이해하기가 어렵다.  
> 이런 현상은 합성곱 층을 많이 쌓을수록 심해진다. 이는 얕은 층의 합성곱 층에서는 이미지의 시각적인 정보를 감지하고,  
> 깊은 층의 합성곱 층에서는 앞에서 감지한 시각적 정보를 바탕으로 추상적인 정보를 학습한다고 볼 수 있다.  
> 즉 자세하고 국소적인 패턴의 학습에서 전반적·추상적 패턴을 학습한다고 볼 수 있다.





