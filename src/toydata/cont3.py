'''
toy data 3:
	- continuous values
	- binary classificarion
	- circular pattern
	- balanced data
'''

import math
import numpy as np
import matplotlib.pyplot as plt

num_samples = 200
radius = 5

# inner cluster
r1 = np.random.uniform(0, radius*0.5, num_samples)
theta1 = np.random.uniform(0, 2*math.pi, num_samples)
x1 = r1 * np.sin(theta1)
y1 = r1 * np.cos(theta1)

# outer cluster
r2 = np.random.uniform(radius*0.7, radius, num_samples)
theta2 = np.random.uniform(0, 2*math.pi, num_samples)
x2 = r2 * np.sin(theta2)
y2 = r2 * np.cos(theta2)

x = np.hstack((x1, x2))
y = np.hstack((y1, y2))
label = ['royalblue'] * num_samples + ['orange'] * num_samples

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=50)
plt.axis('off')

plt.savefig('d3.png', transparent=True)