## 핵심 키워드
> **로지스틱 회귀(logistic regression)**: 선형 방정식을 이용한 분류 알고리즘.
> 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 이용해 클래스 확률을 출력할 수 있다.  
> <img width="300" height="200" alt="image" src="https://github.com/user-attachments/assets/13643a92-7a88-435e-84f6-e9821c9fdbe7" />
><img width="300" height="120" alt="image" src="https://github.com/user-attachments/assets/1e3fa593-21cc-4381-8d02-3a0aff102b3c" />

> **다중 분류**: 타깃 클래스가 2개 이상인 분류 문제. 로지스틱 회귀는 다중 분류를 위해 소프트 맥스 함수를 이요해 클래스를 예측한다.

> **시그모이드 함수(sigmoid function)**: 로지스틱 함수라고도 부르며, 선형 방정식의 값을 0과 1 사이의 값으로 압축한다.
> 이진 분류(타깃 클래스가 2개인 분류 문제)를 위해 사용된다.

> **소프트맥스 함수(softmax function)**: 다중 분류에서 여러 선형 방정식의 출력 결괄르 정규화하여 합이 1이 되도록 만든다.
> 로지스틱 회귀 알고리즘은 다중 분류 시 모든 개별 클래스의 Z값(표준점수)를 산출하고, 그 중 가장 높은 값을 가진 클래스를 정답으로 예측한다.


## 핵심 패키지와 함수
### scikit-learn
> **LogisticRegression**: 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스이다.
> -solver 매개변수에서 사용할 알고리즘을 사용할 수 있다. sag, saga, newton-cholesky(대규모 데이터셋에 효율적) 등이 있다.  
> -penalty 매개변수에서
