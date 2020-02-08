from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

data = make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=1)
X, y = data
X = pd.DataFrame(X, columns=['x', 'y'])
y = pd.DataFrame(y, columns=['labels'])
data = pd.concat([X, y], axis=1)

# step 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X['x'], X['y'], c=y['labels'], cmap=plt.cm.RdBu)

# step 3
#circle = plt.Circle((0, 0), 0.75, color='black', fill=False)
#ax.add_artist(circle)

# step 2
#z = 2*np.power(X['x'], 2) + 2*np.power(X['y'], 2)
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X['x'], X['y'], z, c=y['labels'], cmap=plt.cm.RdBu)

#xx, yy = np.meshgrid(range(-1,2), range(-1,2))
#z = np.ones([3,3])
#ax.plot_surface(xx, yy, z, alpha=0.2)

plt.show()