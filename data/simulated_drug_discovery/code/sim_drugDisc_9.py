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

X = pd.DataFrame(X, columns=['chemical_feature_1','chemical_feature_2'])
y = pd.DataFrame(y, columns=['class'])
data = pd.concat([X, y], axis=1)
data = data.sample(frac=1)

plt.figure(figsize=(4,4), dpi=100)
plt.scatter(X['chemical_feature_1'], X['chemical_feature_2'], c=y['class'], cmap=plt.cm.brg)
plt.axis('off')
plt.savefig('../images/simulated_drug_discovery.png', transparent=True)

data['class'] = data['class'].replace(0,'non-interacting')
data['class'] = data['class'].replace(1,'interacting')
data.to_csv('../simulated_drug_discovery.csv', sep=',', index=False)
#plt.show()

