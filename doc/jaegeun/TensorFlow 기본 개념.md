# TensorFlow 기본 개념
TensorFlow Machine Learning Cookbook을 읽기 전 TensorFlow에 대한 이해가 어느정도 있다는 가정하게 글이 시작되어 TensorFlow에 대한 간략한 내용부터 정리를 시작합니다.

### TensorFlow 기본 개념

To use TensorFlow you need to understand how TensorFlow:
- Represents computations as graphs.
    - 그래프로 데이터 계산을 구조화 한다.
- Executes graphs in the context of Sessions.
    - Session 이라는 개념으로 그래프를 실행한다.
- Represents data as tensors.
    - 데이터(Data, Edge, Data array 등 여러가지로 불림)를 Tensor로 표현한다.
    - 텐서(Tensor)는 배열(array)로서 어떤 값이든 될 수 있다.
- Maintains state with Variables.
    - 데이터의 상태를 변수로 저장한다.
- Uses feeds and fetches to get data into and out of arbitrary operations.
    - 임의의 연산에서 Feed와 Fetch를 사용해 데이터를 꺼내오거나 저장한다.

### TensorFlow 용어 해설 (사실 이해를 해야 하는 부분이라 여러내용들을 전부 가져옴)
1. 첫번째 [# 링크](http://www.jinniahn.com/2016/07/blog-post_28.html)
    - 오퍼레이션 : 데이터를 받아서 처리하는 명령어. 데이터를 더하거나 빼거나 하는 명령어들이 있다.
    - 텐서 : 데이터, 텐서는 데이터를 담고 있는 배열 같은 것
    - 세션 : 실행환경, 오퍼레이션을 동작시키기 위해서 세션이 필요하다.
    - 변수 : 오퍼레이션이 실행되면서 데이터가 변경되는데 이것을 담는 통
2. 두번째 [# 링크](http://forensics.tistory.com/5)
    - 텐서 : n차원 행렬(matrix)
    - 점(node) : 그래프에서 수학 연산을 나타내며 선을 통해 입력받은 데이터를 연산하여 결과를 선으로 출력한다.
    - 선(edge) : 점들을 지나다니는 다차원 데이터 배열(multidimensional data array)로 텐서(tensor)라고 함
    - 데이터 흐름 그래프(data flow graph) : 데이터들이 점을 지나면서 연산이 이루어지고 결국에는 내가 원하는 결과를 얻거나 작업이 이루어지는 과정으로 텐서가 그래프 상에서 돌아다닌다., 즉 텐서 플로우(tensor flow)이다.
3. 세번째 [# 링크 -> 여긴 진짜 들어가서 전부 읽어보는거 추천](http://sanghyukchun.github.io/57/)
    - 조금 더 엄밀하게 Machine Learning을 정의해보자. Machine Learning problem은 아래 요소들로 구성이 된다.
        - Experience E를 Learning할 Computer Program
        - 각각의 E에 대응되는 class of task T
        - Task의 Performace Measure P
    - 위에서 기술한 세 가지 요소들로부터 Machine Learning problem을 다음과 같이 정의할 수 있다. ‘Experience 'E'를 사용하여 (Learning하여) task 'T'의 performance 'P'가 개선이 되도록 하는 program (Algorithm)’. 다시 말해서 머신러닝 문제는 어떤 task 'T'를 풀기 위한 알고리즘을 개발하는 분야인데 이 알고리즘의 performance measure가 ‘기계’ 혹은 Compute program이 ‘Learning’할 수 있는 Experience 'E'를 사용해 개선될 수 있는 알고리즘을 개발하는 문제인 것이다. 스팸필터 문제로 돌아가보자. 스팸필터 문제에서 해결하고자 하는 Task는 새로운 메일을 받았을 때 해당 메일이 스팸메일인지 아니면 그렇지 않은지 판별하는 것이다. 따라서 algorithm의 performance measure 'P'는 얼마나 정확하게 스팸을 골라냈는지 accuracy를 측정하면 간단하게 구할 수 있다. 또한 Experiment 'E'는 예전에 받았던 이메일 들이 될 것이다. 또한 블랙리스트 등은 데이터가 많다고 해서 그 성능이 개선되는 것이 아니기 때문에 머신러닝이라고 할 수 없다. 우리가 하고 싶은 것은, 스팸인지 아닌지 판별하기 위해 예전 이메일들로 스팸 필터를 ‘학습’ 시켜서 스팸 필터의 성능을 향상시키는 것이다.
    - Machine Learnig이 하는 일은 주어진 ‘데이터’ X=(x1,x2,x3,…,xn)X=(x1,x2,x3,…,xn)와 각 데이터에 대응하는 실제 ‘현상’ Y=(y1,y2,…,yn)Y=(y1,y2,…,yn)에 대한 ‘관계’ function 'f'를 찾는 과정과 같다. 정확한 함수 'f'를 찾기 위해 Machine Learnig 알고리즘들은 데이터에 대한 가정을 하고, 그 가정에 따라 주어진 데이터를 ‘최대한 잘 설명할 수 있는’, 함수 'f′'을 찾는다. 이때 이런 'f′'을 Hypothesis라고 한다.

### 일반 소프트웨어와 ML 소프트웨어의 차이점???, 일반적인 비교

1. ML(Machine Learning)은 소프트웨어 이다.
    - 일반적으로 알고있는 소프트웨어, 즉 프로그램들은 명시적인 프로그램(Explicit programs)이다.
        - 개발자가 정한 룰안에서 룰에 따라 반응하는 프로그램
    - ML 프로그래밍 방식이 아닌 명시적인 프로그래밍의 한계
        - 룰이 많아지는 경우 명시적으로 프로그래밍하기에 한계가 있음
        - 예를들어, 스팸 메일을 검출해 내는 메일의 경우 시시때때로 바뀌는 스팸메일을 패턴을 적용할 수 없으며, 자율 주행과 같이 예외적인 상황이 빈번한 프로그램을 대상으로 명시적 프로그래밍을 적용 할 수 없음.

2. ML에서 학습(Learning)이란?
    - ML을 위해서는 반드시 Data Set(Training data set, data ararry 등으로 불림) 이 필요
    - ML에서 학습이란 2가지로 분류된다.
        - Supervised Learning (해석하자면 지도 학습, 감독하에 진행되는 학습 등?)
        - Unsupervised Learning (그렇지 않은 학습 ㅋㅋ)

### Supervised Learning
학습에 사용되는 데이터에 결과가 정해졌 있는 경우를 Supervised Learning이라고 한다.
예를들어 공부 시간과 성적에 대한 결과의 데이터셋이 있고 그를 바탕으로 결과를 얻는 학습을 이야기한다.

 |공부 시간| 점수 |
 |:------:|:----:|
 | 1 | C |
 | 1 | D |
 | 1 | C |
 | 2 | A |
 | 2 | A+ |
 | 3 | A |

라는 데이터가 주어졌을때 눈으로 보기에도 공부 시간이 늘면 점수가 높다는 것을 알 수가 없을 것이다. 이런것 처럼 기본 데이터셋(Training data set)이라고 하며 '공부 시간'과 같은 조건을 Feature 또는 Attribute 라고 하며, Feature에 따라 결정되는 값을 Targeted value(목적값)이라고 한다.

### Regression problem vs Classification problem
 위에서 언급한 Supervised Learning의 결과값에 따라 Regression과 Classification 나뉘게 된다.
 - Regression
    - Predict continuous valued output
    - 지속적인 값에대한 결과를 얻는 방식
 - Classification
    - Discrete valued output (0 or 1)
    - pass/non-pass, true/false, 1/0 와 같이 고정된 결과
 - Multi Label Classification
    - 등급(A~F등급)과 같이 레이블을 통해 결과를 구간별로 나뉘어지는 결과 

### Unsupervised learning
Supervised Learning과 반대의 개념으로 생각하면 된다. 데이터에 대한 attribute는 있지만 주어진 정답(Label)이 없는 경우를 말한다.\
예를 들어 데이터에 대한 여러가지의 집단이 분류되 었을대, 각 집단에 대해 정의를 내려주지 않는 방식을 이야기한다. 집단에 대한 정의를 내려주는 경우 Supervised Learning이 되어버린다.

 