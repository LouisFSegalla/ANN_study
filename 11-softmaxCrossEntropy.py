import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)


def cross_entropy(actual,predicted):
    loss = -np.sum(actual*np.log(predicted))
    return loss

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy = ', outputs)


x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x,dim=0)
print('softmax torch = ', outputs)


Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad  = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')


loss = nn.CrossEntropyLoss()

Y = torch.tensor([2,0,1])
#n_samples x n_class = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1],[2.0, 3.0, 0.1]])
Y_pred_bad  = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1],[0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(f'Loss1 torch: {l1.item():.4f}')
print(f'Loss2 torch: {l2.item():.4f}')

_, predictions1 = torch.max(Y_pred_good,1)
_, predictions2 = torch.max(Y_pred_bad,1)

print(predictions1)
print(predictions2)
