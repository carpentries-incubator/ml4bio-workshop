'''
toy data 4:
	- continuous values
	- binary classificarion
	- unseparable data
	- unbalanced data
'''

import numpy as np
import matplotlib.pyplot as plt

num_samples1 = 1000
num_samples2 = 100

cluster1 = 1.5 * np.random.randn(num_samples1, 2)
cluster2 = 0.5 * np.random.randn(num_samples2, 2) + [2.5, 2.5]
data = np.vstack((cluster1, cluster2))
label = ['orange'] * num_samples1 + ['royalblue'] * num_samples2

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data[:,0], data[:,1], c=label, s=50)
plt.axis('off')

plt.savefig('d4.png', transparent=True)

