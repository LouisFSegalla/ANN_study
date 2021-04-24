
import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------------- #

def read_data():

	df = pd.read_excel("dadosTrabalhoRNA.xlsx",index_col=0)
	list_values = df.values.tolist()

	x = []
	y = []

	for item in list_values: # dados de entrada e saida
		x.append([item[0]])
		y.append([item[1]])

	return x,y


def train_net(inputs,outputs):

	input_size = 1
	hidden_layers_dimensions = [12,12,10] # numero de unidade ocultas em cada camada oculta (3 camadas ocultas)
	output_size = 1

	model = torch.nn.Sequential(torch.nn.Linear(1,hidden_layers_dimensions[0]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[0],hidden_layers_dimensions[1]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[1],hidden_layers_dimensions[2]),
								torch.nn.ReLU(),
								torch.nn.Linear(hidden_layers_dimensions[2],output_size))

	learning_rate = 1e-4
	epochs = 35000

	loss_fn = torch.nn.MSELoss(reduction="sum")
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for t in range(epochs):
		y_pred = model(inputs)
		loss = loss_fn(y_pred,outputs)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if t % 250 == 249:
			print(t,loss.item())

	return model

# --------------------------------------------------------------------------------------------------------------------------------------------- #

# Lendo dados do arquivo .csv
x,y = read_data()

number_samples_total = len(x)
number_samples_test = int(0.2*number_samples_total)

list_test_values = []

# Removendo 20% de itens dos vetores de entrada e saida do conjunto de treino
for i in range(number_samples_test):
	
	index = random.randint(0,len(x)-1)
	list_test_values.append(x.pop(index)) # Removendo valor do conjunto de treino e o adicionando ao de teste
	y.pop(index) # Removendo valor de y do conjunto de treino

print(list_test_values)

tensor_x = torch.FloatTensor(x)
tensor_y = torch.FloatTensor(y)

# Treinamento da rede
model = train_net(tensor_x,tensor_y)

# Listas dos valores de entrada e saida (valores preditos pela rede)
inputs = []
outputs = []

# Estimativas a partir da rede treinada
for i in range(number_samples_total):
	test_value = torch.tensor([i],dtype=torch.float32)
	inputs.append(i)
	outputs.append(model(test_value).item())

# Plotando pontos do conjunto de teste
'''
inputs = list_test_values
for item in inputs:
	test_value = torch.tensor([item],dtype=torch.float32)
	outputs.append(model(test_value).item())	
'''

# Plot dos graficos

plt.figure()
plot_orig, = plt.plot(x,y)
plot_pred, = plt.plot(inputs,outputs,"ro")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Curvas original e valores estimados função")
plt.legend([plot_orig,plot_pred],["Curva Original","Valores preditos pela rede"])
plt.show()