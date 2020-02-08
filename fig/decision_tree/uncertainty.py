import numpy as np
import matplotlib.pyplot as plt

y = 1
z = np.arange(0.005, 0.995, .005)
gini = 2*z*(1-z)
entropy = -z*np.log2(z)-(1-z)*np.log2(1-z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, gini, label='gini')
ax.plot(z, entropy, label='entropy')
ax.set_ylim([0,1])
ax.set_xlim([0,1])
ax.set_xlabel('p(k=1)')
ax.set_ylabel('H(D)')
ax.legend(loc='best')

plt.show()