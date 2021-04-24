import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 0) prepare data
filename = "dados.txt"

X_numpy, Y_numpy = np.loadtxt(filename,usecols=(0,1),unpack=True)


X_train, X_test, Y_train, Y_test = train_test_split(X_numpy,Y_numpy,test_size=0.2, random_state=1234)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test  = torch.from_numpy(Y_test.astype(np.float32))


X_train = X_train.view(X_train.shape[0],1)
X_test = X_test.view(X_test.shape[0],1)
Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)


#1) model
#hyper parameters
input_size = 1
hidden_size = 10
output_size = 1
num_epoch = 35000
learning_rate = 1e-4

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(NeuralNet,self).__init__()
        self.linearIn  = nn.Linear(input_size,hidden_size)
        self.relu1     = nn.ReLU()
        self.linear1   = nn.Linear(hidden_size,hidden_size)
        self.relu2     = nn.ReLU()
        self.linear2   = nn.Linear(hidden_size,hidden_size)
        self.relu3     = nn.ReLU()
        self.linear3   = nn.Linear(hidden_size,hidden_size)
        self.relu4     = nn.ReLU()
        self.linearOut = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = self.linearIn(x)
        out = self.relu1(out)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.linear2(out)
        out = self.relu3(out)
        out = self.linear3(out)
        out = self.relu4(out)
        out = self.linearOut(out)
        return out


model = NeuralNet(input_size,hidden_size,output_size)

#loss and optimizer
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#training loop
n_total_steps = len(X_train)

for epoch in range(num_epoch):
    #forward pass
    outputs = model(X_train)
    loss    = criterion(outputs,Y_train)

    #backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1) % 1000 == 0:
        print(f'epoch={epoch+1}/{num_epoch},loss = {loss.item():.4f}')

#test
with torch.no_grad():
    n_correct = 0
    n_samples = Y_test.shape[0]

    y_predicted = model(X_test)
    for i in range(len(X_test)):
        print(f'|{y_predicted[i]} - {Y_test[i]}| = {abs(y_predicted[i]-Y_test[i])}')
        if(abs(y_predicted[i]-Y_test[i]) < 1):
            n_correct += 1


    acc = 100.0*n_correct / n_samples
    print(f'accuracy = {acc}%')

    #X_numpy, Y_numpy
    X_curva = []
    Y_curva = []
    for i in range(len(X_numpy)):
    	test_value = torch.tensor([i],dtype=torch.float32)
    	X_curva.append(i)
    	Y_curva.append(model(test_value).item())

    real,     = plt.plot(X_numpy,Y_numpy,'b-')
    estimada, = plt.plot(X_curva,Y_curva,'r--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Comparação entre curva real e curva estimada")
    plt.legend([real,estimada],["Curva Original","Valores preditos pela rede"])
    plt.grid()
    plt.show()
