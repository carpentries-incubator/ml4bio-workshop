'''
toy data 2:
	- continuous values
	- binary classificarion
	- XOR pattern
	- balanced data
'''

import numpy as np
import matplotlib.pyplot as plt

num_samples = 400

x = np.random.uniform(-3,3,num_samples)
y = np.random.uniform(-3,3,num_samples)
padding = 0.2

# leave some space to ensure separability
x += ((x>0)-0.5) * 2 * padding
y += ((y>0)-0.5) * 2 * padding
z = (x*y >= 0)

label = []
for i in range(0, np.shape(z)[0]):
	if z[i]:
		label.append('orange')
	else:
		label.append('royalblue')

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=50)
plt.axis('off')

plt.savefig('d2.png', transparent=True)