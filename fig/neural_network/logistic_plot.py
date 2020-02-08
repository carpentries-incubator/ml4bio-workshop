import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, .1)
t = 1/(1+np.exp(-z))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axhline(0, color='lightgrey')
plt.axvline(0, color='lightgrey')
ax.plot(z, t)
ax.set_ylim([-1,1])
ax.set_xlim([-5,5])
ax.set_xlabel('z')
ax.set_ylabel('f(z)')
ax.set_title('logistic')

plt.show()