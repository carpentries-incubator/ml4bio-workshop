import sys, plotSVM
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import svm

import random

class App(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.setWindowTitle('matplotlib plotting test')
		self.setGeometry(300, 300, 600, 600)

		m = MPLCanvas(self, width=6, height=6, dpi=100)
		m.move(0, 0)
		self.show()

class MPLCanvas(FigureCanvas):
	def __init__(self, parent, width, height, dpi):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(1, 1, 1)

		FigureCanvas.__init__(self, fig)
		self.setParent(parent)

		#FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		#FigureCanvas.updateGeometry(self)

		with open('iris.csv', 'r') as f:
			iris = pd.read_csv(f, delimiter=',', header=None)

		features = iris.iloc[:,:2]
		label = iris.iloc[:,4]
		label = pd.factorize(label)[0]

		model = svm.SVC(kernel='linear')
		fit = model.fit(features, label)

		title = 'SVM with linear kernel'
		x = features.iloc[:,0]
		y = features.iloc[:,1]
		xx, yy = self.make_meshgrid(x, y)
		ax = self.figure.add_subplot(1, 1, 1)

		self.plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

		ax.scatter(x, y, c=label, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xlabel('Sepal length')
		ax.set_ylabel('Sepal width')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title)

		self.draw()

	def make_meshgrid(self, x, y, h =0.02):
		x_min, x_max = x.min() - 1, x.max() + 1
		y_min, y_max = y.min() - 1, y.max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		return xx, yy

	def plot_contours(self, ax, model, xx, yy, **params):
		Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx, yy, Z, **params)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())
