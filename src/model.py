import warnings
import pandas as pd
import numpy as np
from sklearn import tree, ensemble, neighbors, linear_model, neural_network, svm, naive_bayes
from sklearn import preprocessing, model_selection, exceptions
from PyQt5.QtWidgets import QTreeWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from model_metrics import ModelMetrics

class Model:
	counter = 0			# number of existing classifiers

	# constructor:
	#
	# classifier:	classifier
	# X:			training features
	# y: 			training labels
	# val_method:	validation method
	# val_size: 	percent of validation data (only used when val_method='holdout')
	# k:			number of folds (only used when val_method='cv')
	# stratify:		stratify samples or not
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		Model.counter += 1

		self.classifier = classifier
		self.X = X
		self.y = y

		# default name for the classifier
		self.name_ = 'classifier_' + str(Model.counter)
		self.type_ = ''
		self.comment_ = ''
		self.params_ = classifier.get_params()	# hyperparameters of classifier

		if val_method == 'holdout':
			self.train_metrics, self.val_metrics = self.__hold_out_validation(\
				classifier, X, y, val_size, stratify)
		elif val_method == 'cv':
			self.train_metrics, self.val_metrics = self.__cross_validation(\
				classifier, X, y, k, stratify)
		elif val_method == 'loo':
			self.train_metrics, self.val_metrics = self.__cross_validation(\
				classifier, X, y, X.shape[0])

		self.test_metrics = None

	# validation using held-out data
	#
	# classifier:	classifier for validation
	# X:			training features
	# y: 			training labels
	# val_size: 	percent of validation data
	# stratify: 	stratify samples or not
	def __hold_out_validation(self, classifier, X, y, val_size, stratify=True):
		if not stratify or val_size == 0:
			s = None
		else:
			s = y

		X_train, X_val, y_train, y_val = model_selection.train_test_split(\
			X, y, test_size=val_size, stratify=s, random_state=0)

		# catch convergence warning
		with warnings.catch_warnings():
			warnings.filterwarnings('error', category=exceptions.ConvergenceWarning)
			try:
				classifier = classifier.fit(X_train, y_train)
			except exceptions.ConvergenceWarning:
				raise

		y_train_pred = classifier.predict(X_train) 			# class prediction
		y_train_prob = classifier.predict_proba(X_train)	# probability of each class

		if X_val.shape[0] != 0:
			y_val_pred = classifier.predict(X_val)
			y_val_prob = classifier.predict_proba(X_val)

			return ModelMetrics(classifier, y_train, y_train_pred, y_train_prob, 'holdout'), \
				ModelMetrics(classifier, y_val, y_val_pred, y_val_prob, 'holdout')
		else:
			return ModelMetrics(classifier, y_train, y_train_pred, y_train_prob, 'holdout'), None

	# k-fold cross validation
	#
	# classifier: 	classifier for Validation
	# X:			training features
	# y:			training labels
	# k: 			number of folds
	# stratify: 	stratify samples or not
	def __cross_validation(self, classifier, X, y, k, stratify=True):
		if k == X.shape[0]:		# leave-one-out
			kf = model_selection.KFold(n_splits=k)
		else:
			if stratify:
				kf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
			else:
				kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)

		# training data and predictions for each fold
		y_train_list = []
		y_train_pred_list = []
		y_train_prob_list = []
		y_val_list = []
		y_val_pred_list = []
		y_val_prob_list = []

		for train_idx, val_idx in kf.split(X, y):
			X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
			y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
			y_train_list.append(y_train)
			y_val_list.append(y_val)

			# catch convergence warning
			with warnings.catch_warnings():
				warnings.filterwarnings('error', category=exceptions.ConvergenceWarning)
				try:
					classifier = classifier.fit(X_train, y_train)
				except exceptions.ConvergenceWarning:
					raise

			y_train_pred_list.append(classifier.predict(X_train))
			y_val_pred_list.append(classifier.predict(X_val))
			y_train_prob_list.append(classifier.predict_proba(X_train))
			y_val_prob_list.append(classifier.predict_proba(X_val))

		if k == X.shape[0]:		# leave-one-out
			y_val = np.hstack(y_val_list)
			y_val_pred = np.hstack(y_val_pred_list)
			y_val_prob = np.vstack(y_val_prob_list)

			return ModelMetrics(classifier, y_train_list, y_train_pred_list, y_train_prob_list, 'cv'), \
				ModelMetrics(classifier, y_val, y_val_pred, y_val_prob, 'loo')
		else:
			return ModelMetrics(classifier, y_train_list, y_train_pred_list, y_train_prob_list, 'cv'), \
				ModelMetrics(classifier, y_val_list, y_val_pred_list, y_val_prob_list, 'cv')

	# test a classifier
	#
	# X: 	test features
	# y: 	test labels
	def test(self, X, y):
		self.test_X = X
		self.test_y = y

		classifier = self.classifier.fit(self.X, self.y)
		y_pred = classifier.predict(X) 			# class prediction
		y_prob = classifier.predict_proba(X)	# probability of each class
		self.test_metrics =  ModelMetrics(classifier, y, y_pred, y_prob, 'holdout')

	# use a classifier to make prediction
	def predict(self, X, path):
		classifier = self.classifier.fit(self.X, self.y)
		y_pred = pd.DataFrame(classifier.predict(X), columns=['prediction'])
		y_prob = pd.DataFrame(np.around(classifier.predict_proba(X), decimals=4), columns=classifier.classes_)
		output = pd.concat([X, y_pred, y_prob], axis=1)
		output.to_csv(path, sep=',', index=False) 		# save to file

	# set classifier name (str)
	def set_name(self, name):
		self.name_ = name

	# set comment on classifier (str)
	def set_comment(self, comment):
		self.comment_ = comment

	# return classifier name (str)
	def name(self):
		return self.name_

	# return classifier type (str)
	def type(self):
		return self.type_

	# return comment on classifier (str)
	def comment(self):
		return self.comment_

	# return classifier hyperparameters (dict)
	def params(self):
		return self.params_

	# return perfermance metrics on training data (ModelMetrics)
	def metrics(self, option):
		if option == 'train':	return self.train_metrics
		elif option == 'val':	return self.val_metrics
		elif option == 'test':	return self.test_metrics

	# show summary of classifier for model selection
	def summary(self, view):
		view.clear()

		name_item = QTreeWidgetItem(view)
		name_item.setText(0, 'Name')
		name_item.setText(1, self.name())
		name_item.setToolTip(1, self.name())
		type_item = QTreeWidgetItem(view)
		type_item.setText(0, 'Type')
		type_item.setText(1, self.type())
		params_item = QTreeWidgetItem(view)
		params_item.setText(0, 'Parameters')
		
		# show classifier hyperparameters
		params = self.params()
		for param in params:
			param_item = QTreeWidgetItem()
			param_item.setText(0, param)
			param_item.setToolTip(0, param)
			value = params[param]
			if value == None:
				param_item.setText(1, 'None')
			else:
				param_item.setText(1, str(value))
			params_item.addChild(param_item)

		metrics_item = QTreeWidgetItem(view)
		metrics_item.setText(0, 'Performance')

		# show metrics on training data
		train_metrics_item = QTreeWidgetItem(metrics_item)
		train_metrics_item.setText(0, 'Training')
		train_metrics = self.metrics('train')
		train_metrics.summary(train_metrics_item)

		# show metrics on validation data
		val_metrics_item = QTreeWidgetItem(metrics_item)
		val_metrics_item.setText(0, 'Validation')
		val_metrics = self.metrics('val')
		if val_metrics is not None:
			val_metrics.summary(val_metrics_item)

		# show metrics on test data
		test_metrics = self.metrics('test')
		if test_metrics is not None:
			test_metrics_item = QTreeWidgetItem(metrics_item)
			test_metrics_item.setText(0, 'Test')
			test_metrics.summary(test_metrics_item)

		comment_item = QTreeWidgetItem(view)
		comment_item.setText(0, 'Comment')
		comment_text_item = QTreeWidgetItem(comment_item)
		comment_text_item.setFirstColumnSpanned(True)
		comment_text_item.setText(0, self.comment())

	# plot decision regions
	#
	# reference: 
	# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
	def plot_decision_regions(self, option, canvas):
		le = preprocessing.LabelEncoder()		# integer encoder
		le.fit(self.y)
		classifier = self.classifier.fit(self.X, le.transform(self.y))

		if option == 'train':
			X = self.X
			y = self.y
		elif option == 'test':
			X = self.test_X
			y = self.test_y

		cm_bkgd = plt.cm.RdBu
		cm_pts = ListedColormap(['#FF0000', '#0000FF'])

		d1 = X.iloc[:, 0] 	# x-axis
		d2 = X.iloc[:, 1]	# y-axis
		d1_slack = (d1.max() - d1.min()) * 0.1
		d2_slack = (d2.max() - d2.min()) * 0.1
		d1_min, d1_max = d1.min() - d1_slack, d1.max() + d1_slack 	# x-axis range
		d2_min, d2_max = d2.min() - d2_slack, d2.max() + d2_slack	# y-axis range
		md1, md2 = np.meshgrid(np.arange(d1_min, d1_max, 0.01), np.arange(d2_min, d2_max, 0.01))

		rcParams.update({'font.size': 6})
		canvas.figure.clear()
		ax = canvas.figure.subplots()
		Z = classifier.predict_proba(np.c_[md1.ravel(), md2.ravel()])[:, 1]
		Z = Z.reshape(md1.shape)
		out = ax.contourf(md1, md2, Z, cmap=cm_bkgd, alpha=0.8)

		ax.scatter(d1, d2, c=le.transform(y), cmap=cm_pts, alpha=0.6, edgecolors='k')
		ax.set_xlim(md1.min(), md1.max())
		ax.set_ylim(md2.min(), md2.max())
		ax.set_xticks(())
		ax.set_yticks(())

		canvas.figure.tight_layout()
		canvas.draw()

class DecisionTree(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)		
		self.type_ = 'decision tree'

class RandomForest(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'random forest'

class KNearestNeighbors(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'k-nearest neighbors'

class LogisticRegression(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'logistic regression'

class NeuralNetwork(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'neural network'

class SVM(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'svm'

class NaiveBayes(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify, mode):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		self.type_ = 'naive bayes'