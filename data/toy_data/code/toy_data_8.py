'''
toy data 8:
	- continuous values
	- three-way classificarion
	- XOR pattern
	- balanced data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_samples = 50
np.random.seed(1)

means = [[-4,2.5], [0,2.5], [4,2.5], [-4,-2.5], [0,-2.5], [4,-2.5]]
cov = [[1,0],[0,1]]

x = np.zeros(num_samples * len(means))
y = np.zeros(num_samples * len(means))

for i in range(0, len(means)):
	new_x, new_y = np.random.multivariate_normal(means[i], cov, num_samples).T
	x[num_samples*i : num_samples*(i+1)] = new_x
	y[num_samples*i : num_samples*(i+1)] = new_y

x = pd.DataFrame(x, columns=['x'])
y = pd.DataFrame(y, columns=['y'])
z = [0] * num_samples + [1] * num_samples + [2] * num_samples + [2] * num_samples + [0] * num_samples + [1] * num_samples
z = pd.DataFrame(z, columns=['labels'])

data = pd.concat([x, y, z], axis=1)
data = data.sample(frac=1)
data.to_csv('../toy_data_8.csv', sep=' ', index=False)

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data['x'], data['y'], c=data['labels'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/toy_data_8.png', transparent=True)
#plt.show()

