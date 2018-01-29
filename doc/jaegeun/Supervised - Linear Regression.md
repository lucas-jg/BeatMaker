# Supervised - Linear Regression (선형 회귀)
Supervised Learning에서 가장 기본적으로 사용되는 Linear Regression에 대한 내용을 다룬다. 먼저 Linear Regression(선형 회귀)에 대한 정의를 알아보자
- 통계학에서, 선형 회귀(線型回歸, 영어: linear regression)는 종속 변수 y와 한 개 이상의 독립 변수 (또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법이다. 한 개의 설명 변수에 기반한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀라고 한다.

- 선형 회귀는 선형 예측 함수를 사용해 회귀식을 모델링하며, 알려지지 않은 파라미터는 데이터로부터 추정한다. 이렇게 만들어진 회귀식을 선형 모델이라고 한다.

 ![img](https://github.com/yunseul-light/BeatMaker/blob/master/img/Linear%20Regressioin.jpg) {.alin}

 *아주 쉽고 간단하게 설명하면 위에 보이는 이미지에서 파란색 점을 대표하는 빨간색 선을 찾아내는 것이 바로 Linear Regression(선형 회귀)이다.*

### Supervised Learning
앞으로 우리가 개발해야하는 목표 또한 Supervised Learning을 기반으로한 프로젝트를 진행하기 때문에 또 다른 설명으로 개념을 한번 설명하고 넘어감.\
명확한 Input과 Output이 존재하며, 훈련을 위한 데이터(데이터셋, 데이터 배열)에 알고리즘(Tensor-Noed)를 적용하여 컴퓨터가 알아서 답을 찾도록 만드는 것을 이야기한다. 

### Regression 
데이터(데이터셋, 데이터 배열)를 대표하는 선형모델을 만들어 그 모델을 통해 주어진 데이터가 어떤 결과를 나타내는지 예측하는 것이다.

### Classification
데이터(데이섯, 데이터 배열)를 대표하는 데이터의 분류를 통해 주어진 데이터가 어떻게 분류되는지 예측하는 것이다.


###  Linear Regression (선형 회귀)

 ![img](https://github.com/yunseul-light/BeatMaker/blob/master/img/Linear%20Regressioin.jpg)

그림을 통해서 아주 간단하게 의미만을 설명
```
 단계 : 데이터 제공(훈련용 데이터) -> 선형 회귀 모델 생성 -> 생성된 선형 회귀 모델을 통해 값 예측
```
1. 훈련용 데이터 제공
 - 데이터를 제공하여 컴퓨터를 훈련시킴
 - 주어진 X(0\~8)에 대한 Y(0\~5) 값을 훈련 데이터로 주어진다.
 - *그래프에서 파란점을 제공해 준다고 생각하면 됨*
2. 주어진 훈련용 데이터에 대한 선영 회귀 분석 모델을 만듬.
 - 주어진 훈련용 데이터를 통해 선영 회귀 모델을 생성
 - *그래프에서 빨간색 선을 만든다고 생각하면 됨*
3. X에 대한 Y값을 결과로 받음
 - 만들어진 선형 회귀 모델에서 주어진 X에대한 Y 값을 결과로 받는다.
 - *그래프에서 보이는 빨간색 선에 X값에 대한 Y값이 얼마인지 찾는다라고 생각하면 됨*

------

[동영상 강의 - 이론](https://www.youtube.com/watch?v=Hax03rCn3UI&index=4&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)\
[동영상 강의 - 실습](https://www.youtube.com/watch?v=mQGwjrStQgg&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=5)

