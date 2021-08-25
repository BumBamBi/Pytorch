# https://wingnim.tistory.com/entry/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EA%B0%95%EC%9D%98-3-Gradient-Descent

import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value
#아무런 숫자가 들어가도 된다.

# model의 forward pass 구현 부분
def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# gradient를 계산
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)

# Before training
print("predict (before training)",  4, forward(4))

# Training loop
# 학습을 시키는 loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        #업데이트 시키는 부분 ! 중요 !!
        w = w - 0.01 * grad 
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
# 학습 완료
print("predict (after training)",  "4 hours", forward(4))