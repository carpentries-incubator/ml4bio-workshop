"""
Binary classification:
Partially overlapping data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

data = make_moons(n_samples=100, noise=0.3, random_state=1)
X, y = data
X = pd.DataFrame(X, columns=['x', 'y'])
y = pd.DataFrame(y, columns=['labels'])
data = pd.concat([X, y], axis=1)
#data.to_csv('../toy_data_1.csv', sep=' ', index=False)

plt.figure(figsize=(4,4), dpi=100)
plt.scatter(X['x'], X['y'], c=y['labels'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/toy_data_1.png', transparent=True)
#plt.show()

