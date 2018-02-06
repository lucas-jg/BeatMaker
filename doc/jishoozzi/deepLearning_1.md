1.1 딥러닝과 텐서플로
 - 딥러닝은 머신러닝 중 '신경망(neural network)'이라는 모델의 일종이다.

1.1.1 머신러닝의 개념
 - 머신러닝은 데이터의 속에 있는 '수학적인 구조'를 컴퓨터로 계산해서 발견해 내는 구조다.
   예를 들어 월별 평균 기온을 예측하기 위하여 x축을 월, y축을 평균기온으로 하여 데이터를 나열하여 연결해보면 구불구불한 직선을 연결한 모습을 연상할 수 있다.
   하지만 직선들을 완만한 곡선으로 바꿔 그려보면 현재 월별 평균기온보다 오차가 발생 할 수 있지만 확률적으로는 바꿔그린 곡선 부근에 다음에 예측할 평균기온이 분포할 가능성이 가장 높다고 할 수 있다.
   이처럼 주어진 데이터의 수치를 있는 그대로 받아들이는 것이 아니라 그 속에 있는 '원리'를 생각하는 것을 '데이터의 모델화'라고 한다. 이렇게 해서 생각해 낸 구조가 바로 '데이터의 모델'인 것이다.
   
   위처럼 데이터의 모델화를 마치고 다음과 같이 과정을 거친다.
   1) 주어진 데이터를 기반으로 해서 미지의 데이터를 예측하는 식을 생각한다.
   2) 식에 포함된 파라미터의 좋고 나쁨을 판단하는 오차 함수를 준비한다.(현재 데이터 값과 과거 데이터들과의 오차)
   3) 오차 함수를 최소화할 수 있도록 파라미터값을 결정한다.

   위 과정에서 예측 정밀도를 향상하기 위해서는 최적의 모델도 필요하지만 대량의 데이터(빅데이터)에 대해 오차 함수를 최소화하는 계산을 할 필요가 있다.
   이 부분을 정해진 알고리즘을 이용해 자동으로 계산하는 것이 머신러닝에서 컴퓨터(머신)가 하는 역할이며 텐서플로의 주요 업무다.
   -> 컴퓨터의 주요역할은 그 식에 포함된 파라미터를 최적화하는 것이다.

1.1.2 신경망의 필요성
   잠시 이거 표현방식을 생각해서 올리겠음 복잡미묘하네