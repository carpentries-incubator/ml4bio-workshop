'''
Binary classification:
Unbalanced data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_samples1 = 200
num_samples2 = 20
np.random.seed(1)

cluster1 = 1.5 * np.random.randn(num_samples1, 2)
cluster2 = 0.5 * np.random.randn(num_samples2, 2) + [2.5, 2.5]
z1 = [0] * num_samples1
z2 = [1] * num_samples2
X = np.vstack((cluster1, cluster2))
z = np.hstack((z1, z2))

X = pd.DataFrame(X, columns=['cell_size','total_intensity'])
z = pd.DataFrame(z, columns=['class'])
data = pd.concat([X, z], axis=1)
data = data.sample(frac=1)

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data['cell_size'], data['total_intensity'], c=data['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_t_cells_7.png', transparent=True)

data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')
data.to_csv('../simulated_t_cells_7.csv', sep=' ', index=False)
#plt.show()



