"""
Binary classification:
Circular pattern
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt

data = make_circles(n_samples=400, noise=0.2, factor=0.05, random_state=1)
X, y = data

rec = []
for i in range(len(y)):
	if len(rec) >= 20:
		break
	if y[i] == 1:
		rec.append(i)

X = np.concatenate((X[np.where(y==0)], X[rec]), axis=0)
y = np.array([0]*200+[1]*20)

X = pd.DataFrame(X, columns=['cell_size','total_intensity'])
y = pd.DataFrame(y, columns=['class'])
data = pd.concat([X, y], axis=1)
data = data.sample(frac=1)

plt.figure(figsize=(4,4), dpi=100)
plt.scatter(X['cell_size'], X['total_intensity'], c=y['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_t_cells_9.png', transparent=True)

data['class'] = data['class'].replace(0,'quiescent')
data['class'] = data['class'].replace(1,'activated')
data.to_csv('../simulated_t_cells_9.csv', sep=',', index=False)
#plt.show()

