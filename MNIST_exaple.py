# https://wikidocs.net/61046
 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits

# 이미지 데이터 로드
digits = load_digits() 

# 첫번째 이미지 배열로 출력 (배열을 보면 값0이 아닌 부분만 보면 모양이 0처럼 보임)
print(digits.images[0])
# 실제 어떤 숫자였는지 확인
print(digits.target[0])
# 전체 샘플 개수
print('전체 샘플의 수 : {}'.format(len(digits.images)))

# 앞에서부터 10개의 레이블 확인
for i in range(10):
  print(i,'번 인덱스 샘플의 레이블 : ',digits.target[10])

# 앞에서부터 10개의 샘플 이미지 출력하기
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]): # 10개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
plt.show()


# 8x8행렬을 64차원으로 저장..? 후 출력 뭔소리지
print(digits.data[0])

# 이미지. 즉, 특성 행렬
X = digits.data 
# 각 이미지에 대한 레이블
Y = digits.target 


##############################################################################
# 다층 퍼셉트론 분류기 만들기

import torch
import torch.nn as nn
from torch import optim

# 모델 생성
model = nn.Sequential(
    nn.Linear(64, 32), # input_layer = 64, hidden_layer1 = 32
    nn.ReLU(),
    nn.Linear(32, 16), # hidden_layer2 = 32, hidden_layer3 = 16
    nn.ReLU(),
    nn.Linear(16, 10) # hidden_layer3 = 16, output_layer = 10
)

# X(이미지)와 Y(레이블) 를 float/int 로 변환
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

# loss구하고 optimizer 선택
# CrossEntropyLoss() 함수는 소프트맥스 함수를 포함하고 있음. >> ??
loss_fn = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters())
losses = []

# 학습
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X) # forwar 연산
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    # 10번의 학습마다 출력
    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, 100, loss.item()))

    losses.append(loss.item())

# loss 출력
plt.plot(losses)
plt.show()