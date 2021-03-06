# https://wingnim.tistory.com/entry/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EA%B0%95%EC%9D%98-6-Logistic-Regression

import torch
import torch.nn.functional as F

# 모델 생성
class MyModel(torch.nn.Module):
    
    def __init__(self):
        super(MyModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
        
    def forward(self,x):
        # sigmoid -> 0 이상이면 1, 음수면 -1이 됨
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# 공부시간(x_data)
x_data = torch.Tensor( [ [1.0],[2.0],[3.0],[4.0] ])
# y_data가 뭔지 모르겠습니다...... 그냥 sifmoid 쓰려고 0과 1을 넣은듯?
y_data = torch.Tensor( [ [0.],[0.],[1.],[1.] ])

# Loss를 구하고 Optimizer을 선택하기
# 모델 객체 생성
model = MyModel()
# Binary Cross Entropy Loss 로 변경
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)



# 트레이닝 사이클 돌리기
for epoch in range(1000):
    
    y_pred = model(x_data)
    
    loss = criterion(y_pred,y_data)
    if(epoch%100==0):
        print(epoch, loss.data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 
hour_var = torch.Tensor([[1.0]])
print("1hour : ",1.0, model(hour_var).data[0][0]>0.5)
hour_var = torch.Tensor([[7.0]])
print("7hour : ",7.0, model(hour_var).data[0][0]>0.5)