# https://wingnim.tistory.com/entry/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EA%B0%95%EC%9D%98-4-Backpropagation-and-Autograd

import torch
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w=torch.tensor([1.0],requires_grad = True)

def forward(x):
    return x*w


def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)


# Training loop
print("predict (before training)" , 4, forward(4))
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data): #1
        l = loss(x_val, y_val) #2
        l.backward() #3
        print("grad: ", x_val, y_val,w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data #4
        # Manually zero the gradients after updating weights
        w.grad.data.zero_() #5

    print("progress:", epoch, l.data[0])
print("predict (after training)" , 4, forward(4))