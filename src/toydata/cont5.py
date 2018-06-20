'''
toy data 5:
	- continuous values
	- three-way classificarion
	- unseparable data
	- balanced data
'''

import pandas as pd
import matplotlib.pyplot as plt

with open('iris.csv', 'r') as f:
	iris = pd.read_csv(f, delimiter=',', header=None)

x = iris.iloc[:, 0]
y = iris.iloc[:, 1]
species = list(iris.iloc[:,4])

label = []
for i in range(0, len(species)):
	if species[i] == 'Iris-setosa':
		label.append('limegreen')
	elif species[i] == 'Iris-versicolor':
		label.append('orange')
	else:
		label.append('royalblue')

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=100)
plt.axis('off')

plt.savefig('d5.png', transparent=True)

