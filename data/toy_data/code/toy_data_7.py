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

X = pd.DataFrame(X, columns=['x', 'y'])
z = pd.DataFrame(z, columns=['labels'])
data = pd.concat([X, z], axis=1)
data = data.sample(frac=1)
#data.to_csv('../toy_data_7.csv', sep=' ', index=False)

fig = plt.figure(figsize=(4,4), dpi=100)
plt.scatter(data['x'], data['y'], c=data['labels'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/toy_data_7.png', transparent=True)
#plt.show()



