# https://wingnim.tistory.com/entry/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EA%B0%95%EC%9D%98-5-Linear-Regression-in-the-PyTorch-way

import torch
from torch.autograd import Variable

x_data = torch.tensor([ [1.0],[2.0],[3.0] ])
y_data = torch.tensor([ [2.0],[4.0],[6.0] ])


# Step 1. variable을 포함하는 class로 디자인 하기

class MyModel(torch.nn.Module):#torch.nn.Module을 상속받음
    #class 생성자 같은 느낌임. 처음 만들어질 때 초기화 해줌. 반드시 필요.
    def __init__(self):
        #부모 class로부터 상속받은 class는 처음 initialize 해줄 때 부모의 init을 해 주어야 한다.
        super(MyModel,self).__init__()

        self.linear = torch.nn.Linear(1,1) #MyModel에 실질적인 연산을 할 모델을 구성하는 부분.
        # 우리는 간단한 모델을 구성할 것이기 때문에 
        # torch API에서 제공하는 torch.nn.Linear만 가지고 우리의 모델을 구성했다.
        # input개수가 한개이고 output 개수도 한개인 Linear model을 만들어주게 된다.
        
    #forward (예측을 수행)하는 함수. 모델을 만들 때 반드시 필요.
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
    
model = MyModel() #model이라는 변수에 만든 모델을 넣어준다.


##################################################################################################################################################


# Step 2. Loss를 구하고 Optimizer을 선택하기

#PyTorch API 로부터 loss 를 만들어내기. (이미 있는 loss들 중 선택할 수 있음.)

#criterion : 기준. 어떤 loss를 줄일 것인지에 대해서 명시한다.
criterion = torch.nn.MSELoss(size_average=False) 
# MSELoss는 Mean Squared Error Loss의 줄임말로, 원본과 예측의 차이의 제곱의 평균을 구해준다는 의미를 가진다.
# size_average를 false로 하면 Loss를 발생시키는 것들의 평균으로 나누지 않는다.


optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
#이전 글에서는 optimizer 없이 직접 그냥 뺀 값을 적용 해 주었는데, 
#이번에는 그 과정 대신 optimizer을 사용한다.
#model.parameters()를 통해서 
#우리가 만든 모델의 파라미터들을 옵티마이저에 전달해주면,
#우리가 이전에 gradient를 사용해서 업데이트하던 
#w = w - grad * learning rate 식 같은 것을 자동으로 해 준다.
#단순히 빼기만 해서 하는게 아닌 SGD(stochastic gradient descent) 라는 
#방법을 써서 optimizing을 진행한다. 자세한건 구글링해보자.
#lr = 0.01로 learning rate를 정해줄 수 있다.


##################################################################################################################################################


# Step 3. 트레이닝 사이클 돌리기

for epoch in range(501):
    # 우리는 모든 x 데이터를 매트릭스(행렬) 형태로 모델에게 넘겨준다.
    y_pred = model(x_data) 
    
    # criterion이라는 함수를 통해서 예측과 정답을 비교하는 평가를 진행한다.
    # 이 때, MSE Loss 를 criterion에 넣었기 때문에 그것을 기준으로 진행하게 된다.
    loss = criterion(y_pred, y_data) 

    if(epoch%100==0):
        print(epoch, loss.data)
    
    #gradient descent 직전에 초기화 해주기.
    optimizer.zero_grad() 

    # 구한 loss로부터 back propagation을 통해 각 변수마다 loss에 대한 gradient 를 구해주기
    loss.backward() 
    
    # step()이란 함수를 실행시키면 우리가 미리 선언할 때 
    # 지정해 준 model의 파라미터들이 업데이트 된다.
    optimizer.step() 
    # 이전 글의 기존 for loop을 이용한 방법으로는 데이터를 
    # 한번에 하나씩 살펴봐야 해서 효율적이지 못했지만 이제는 한번에 묶어서 계산한다.
    # 지금 데이터는 3개라 한번에 봐도 문제가 없지만 몇백만개 이상이 되면 문제가 생긴다.
    # SGD를 통해서 업데이트를 진행할 경우에는 mini - batch를 
    # 사용하는 기법을 통해 이를 해결한다. 자세한 방법은 구글링 해 보자.
