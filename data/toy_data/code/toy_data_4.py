"""
Three-way classification:
Gaussian divided into three quantitles

Reference: http://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_gaussian_quantiles
from matplotlib import pyplot as plt

data = make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=3, random_state=0)
X, y = data
X = pd.DataFrame(X, columns=['x', 'y'])
y = pd.DataFrame(y, columns=['labels'])
data = pd.concat([X, y], axis=1)
#data.to_csv('../toy_data_4.csv', sep=' ', index=False)

plt.figure(figsize=(4,4), dpi=100)
plt.scatter(X['x'], X['y'], c=y['labels'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/toy_data_4.png', transparent=True)
#plt.show()