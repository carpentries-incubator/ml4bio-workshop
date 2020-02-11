'''
Binary classification:
Spiral pattern
'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_samples = 100
np.random.seed(1)

index = np.array(range(num_samples))
r = index/num_samples * 5
t1 = 1.75 * index/num_samples * 2 * math.pi
t2 = 1.75 * index/num_samples * 2 * math.pi + math.pi

x1 = r * np.sin(t1) + np.random.uniform(-0.5, 0.5, num_samples)
y1 = r * np.cos(t1) + np.random.uniform(-0.5, 0.5, num_samples)
z1 = [0] * num_samples

x2 = r * np.sin(t2) + np.random.uniform(-0.5, 0.5, num_samples)
y2 = r * np.cos(t2) + np.random.uniform(-0.5, 0.5, num_samples)
z2 = [1] * num_samples

x = np.hstack((x1, x2))
y = np.hstack((y1, y2))
z = np.hstack((z1, z2))

x = pd.DataFrame(x, columns=['x'])
y = pd.DataFrame(y, columns=['y'])
z = pd.DataFrame(z, columns=['class'])
data = pd.concat([x, y, z], axis=1)
data = data.sample(frac=1)
data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')
data.to_csv('../simulated_t_cells_5.csv', sep=' ', index=False)

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(x['x'], y['y'], c=z['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_t_cells_5.png', transparent=True)
#plt.show()

