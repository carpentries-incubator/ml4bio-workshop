import numpy as np
import matplotlib.pyplot as plt

y = 1
z = np.arange(-0, 1, .02)
t = -y*np.log(z)-(1-y)*np.log(1-z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, t)
ax.set_ylim([0,3])
ax.set_xlim([0,1])
ax.set_xlabel('y_predict')
ax.set_ylabel('cross entropy')
ax.set_title('y_true = 1')

plt.show()