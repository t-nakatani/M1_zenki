#参考: https://note.nkmk.me/python-pytorch-dtype-to/, https://qiita.com/dem_kk/items/66a39a899890a33f9604, http://techno-road.com/blog/?content=7

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# Random initialization
torch.manual_seed(0)
 
# input x, teacher t
data = [[1,1],[1,0],[0,1],[0,0]]
label = [[1,0],[0,1],[0,1],[1,0]]
x = torch.tensor(data, dtype=torch.float32)
y = torch.tensor(label, dtype=torch.float32) 
print("===TRAIN_DATA===")
print("x=",x)
print("y=",y)
 
# dataset
dataset = TensorDataset(x,y)
# data loader
train = dataset
batch_size = 1 # mini batch size
train_loader = DataLoader(train, batch_size, shuffle=True)
 
# My Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        mid = 2 # mid layer
        # input:2 mid:2
        self.fc1 = nn.Linear(2,mid)
        # mid:2 output:1
        self.fc2 = nn.Linear(mid,2)
 
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
# new
net = Net()
 
# Loss function
criterion = nn.MSELoss(reduction="sum")
 
# Stochastic gradient descent
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
 
# Train
print("===TRAIN===")
max_epoch=2000
for epoch in range(max_epoch+1) :
    for batch in train_loader:
        x,y = batch
        #clear grad
        optimizer.zero_grad() 
        #forward
        pred=net(x)
        #loss function
        loss = criterion(pred,y)
        #BP
        loss.backward()
        #update
        optimizer.step()
    if epoch % 500 == 0:
        print("epoc:", epoch, ' loss:', loss.item())

# Test
print("===Test===")
net.eval()
with torch.no_grad() :
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = net(x)
    y_label = torch.argmax(y, dim=1)
    for x_, y in zip(x, y_label):
        print('input:{}, output:{}'.format(x_, y.item()))