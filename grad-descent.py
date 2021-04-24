import numpy as np

#Inicializa o problema
X = np.array([1,2,3,4],dtype=np.float32)

Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0


#modelar a previsão
def forward_pass(x,w):
    return w*x

#perda = MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()


#gradiente
def gradient(x,y,y_predicted):
    return (np.dot(2*x, y_predicted-y)).mean()

print(f'Prediction before the training: f(5) = {forward_pass(5,w):.3f}')

learning_rate = 0.01
n_iters       = 10

for epoch in range(n_iters):
    #Prediction
    y_predicted = forward_pass(X,w)
    #loss
    l           = loss(Y,y_predicted)
    #gradient
    dw          = gradient(X,Y,y_predicted)

    #update the weights
    w -= learning_rate*dw

    if(epoch % 1 == 0):
        print(f'epoch{epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after the training: f(5) = {forward_pass(5,w):.3f}')
