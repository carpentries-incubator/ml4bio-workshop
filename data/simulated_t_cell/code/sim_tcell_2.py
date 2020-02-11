"""
Binary classification:
Circular pattern
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt

data = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=1)
X, y = data
X = pd.DataFrame(X, columns=['cell_size','total_intensity'])
y = pd.DataFrame(y, columns=['class'])
data = pd.concat([X, y], axis=1)
data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')
data.to_csv('../simulated_t_cells_2.csv', sep=',', index=False)

plt.figure(figsize=(4,4), dpi=100)
plt.scatter(X['cell_size'], X['total_intensity'], c=y['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_t_cells_2.png', transparent=True)
#plt.show()

