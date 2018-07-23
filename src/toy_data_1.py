import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

data = make_moons(n_samples=100, noise=0.3, random_state=0)
X, y = data
X = pd.DataFrame(X, columns=['x', 'y'])
y = pd.DataFrame(y, columns=['labels'])
data = pd.concat([X, y], axis=1)
data.to_csv('toy_data_1t.csv', sep=' ', index=False)

plt.figure()
plt.scatter(X['x'], X['y'], c=y['labels'], cmap=plt.cm.RdBu)
plt.show()

