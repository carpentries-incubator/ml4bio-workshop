'''
toy data 1:
	- continuous values
	- binary classificarion
	- linearly separable
	- balanced data
'''

import numpy as np
import matplotlib.pyplot as plt

num_samples = 200

# means and covariance matrices for the generating distribution
mean1 = [-1.5, -1.5]
mean2 = [1.5, 1.5]
cov = [[0.3, 0], [0, 0.3]]

x1, y1 = np.random.multivariate_normal(mean1, cov, num_samples).T
x2, y2 = np.random.multivariate_normal(mean2, cov, num_samples).T
x = np.vstack((x1, x2))
y = np.vstack((y1, y2))
label = ['orange'] * num_samples + ['royalblue'] * num_samples

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=50)
plt.axis('off')

plt.savefig('d1.png', transparent=True)