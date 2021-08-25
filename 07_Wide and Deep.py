# https://wingnim.tistory.com/entry/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EA%B0%95%EC%9D%98-7-Wide-and-Deep

# data link : https://github.com/hunkim/PyTorchZeroToAll

import torch 
from torch.autograd import Variable
import numpy as np 

# 데이터 읽어오기
# 8가지 수치를 보고 당뇨 혼지인지 맞추기
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32) 
# 8가지 수치정보(x_data)
x_data = torch.from_numpy(xy[:, 0:-1])
# 당뇨환자 여부(y_data)
y_data = torch.from_numpy(xy[:, [-1]])


# 모델 클래스 생성
class MyModel(torch.nn.Module):
    
    def __init__(self):
        super(MyModel,self).__init__()
        # (input, output)
        # Layer의 output 다음 Layer의 input과 같아야함
        # 처음의 입력 데이터의 수 8(8가지 수치), 마지막(결과)은 1 이어야함.

        # 그러나 Sigmoid로 Layer를 너무 많이 쌓으면 vanishing gradient 문제 발생
        # sigmoid의 미분값이 0.xx이기 때문에 무한히 곱하면 0에 근사해지는 것
        # 따라서 다른 Acrivation Func를 사용 (주로 Relu)

        # 8개의 수치를 4개의 결과 -> 4개의 결과를 6개의 결과로 -> 6개를 하나의 결과로
        # 왜 8 4 6 1 로 가는 지 모르겠음.. 8 4 4 1로 하면 안되나? 
        # 중간에 수치를 엄청 크게하면 처리속도가 느려진 대신 정확도가 늘었는데 흠....
        self.l1=torch.nn.Linear(8,1)
        self.l2=torch.nn.Linear(1,1231)
        self.l3=torch.nn.Linear(1231,1)
        
        self.sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


# 모델 생성
model = MyModel()

criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# 학습 트레이닝
for epoch in range(1000):
    y_pred = model(x_data)
    
    loss=criterion(y_pred,y_data)
    
    if(epoch%100==0):
        print("loss : ",loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

cnt=0
for it in range(y_data.size()[0]):
    # 0.5 이상은 1, 이하는 0으로 예측
    if( (y_pred[it][0]>0.5)==y_data[it][0].type(torch.ByteTensor)):
        cnt= cnt+1
        
print(cnt)
print("Accuracy : ",cnt*100/y_data.size()[0])