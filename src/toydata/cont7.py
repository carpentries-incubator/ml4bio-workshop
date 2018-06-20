'''
toy data 7:
	- continuous values
	- binary classificarion
	- spiral pattern
	- balanced data
'''

import math
import numpy as np
import matplotlib.pyplot as plt

num_samples = 200
index = np.array(range(num_samples))
r = index/num_samples * 5
t1 = 1.75 * index/num_samples * 2 * math.pi
t2 = 1.75 * index/num_samples * 2 * math.pi + math.pi

x1 = r * np.sin(t1) #+ np.random.uniform(-1, 1, num_samples)
y1 = r * np.cos(t1) #+ np.random.uniform(-1, 1, num_samples)

x2 = r * np.sin(t2) #+ np.random.uniform(-1, 1, num_samples)
y2 = r * np.cos(t2) #+ np.random.uniform(-1, 1, num_samples)

x = np.hstack((x1, x2))
y = np.hstack((y1, y2))

label = ['orange'] * num_samples + ['royalblue'] * num_samples

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x, y, c=label, s=20)
plt.axis('off')

plt.savefig('d7.png', transparent=True)

