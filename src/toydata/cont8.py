'''
toy data 8:
	- continuous values
	- three-way classificarion
	- XOR pattern
	- balanced data
'''

import numpy as np
import matplotlib.pyplot as plt

num_samples = 100

means = [[-4,2.5], [0,2.5], [4,2.5], [-4,-2.5], [0,-2.5], [4,-2.5]]
cov = [[1,0],[0,1]]

x = np.zeros(num_samples * len(means))
y = np.zeros(num_samples * len(means))

for i in range(0, len(means)):
	new_x, new_y = np.random.multivariate_normal(means[i], cov, num_samples).T
	x[num_samples*i : num_samples*(i+1)] = new_x
	y[num_samples*i : num_samples*(i+1)] = new_y

label = ['orange'] * num_samples + ['limegreen'] * num_samples + ['royalblue'] * num_samples \
	 + ['royalblue'] * num_samples + ['orange'] * num_samples + ['limegreen'] * num_samples

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=20)
plt.axis('off')

plt.savefig('d8.png', transparent=True)

