'''
Binary classification:
Sparse data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_samples = 20
np.random.seed(1)

cluster1 = np.random.randn(num_samples, 2) + [1, 1]
cluster2 = np.random.randn(num_samples, 2)
z1 = [0] * num_samples
z2 = [1] * num_samples
X = np.vstack((cluster1, cluster2))
z = np.hstack((z1, z2))

X = pd.DataFrame(X, columns=['cell_size','total_intensity'])
z = pd.DataFrame(z, columns=['class'])
data = pd.concat([X, z], axis=1)
data = data.sample(frac=1)

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data['cell_size'], data['total_intensity'], c=data['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_t_cells_6.png', transparent=True)

data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')
data.to_csv('../simulated_t_cells_6.csv', sep=' ', index=False)
#plt.show()


