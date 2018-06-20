'''
toy data 5:
	- continuous values
	- binary classificarion
	- unseparable data
	- sparse data
	- balanced data
'''

import numpy as np
import matplotlib.pyplot as plt

num_samples = 10

np.random.seed(0)
cluster1 = np.random.randn(num_samples, 2) + [1, 1]
cluster2 = np.random.randn(num_samples, 2)
data = np.vstack((cluster1, cluster2))
label = ['orange'] * num_samples + ['royalblue'] * num_samples

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data[:,0], data[:,1], c=label, s=200)
plt.axis('off')

plt.savefig('d6.png', transparent=True)
