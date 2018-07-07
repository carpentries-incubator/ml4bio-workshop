import pandas as pd
import numpy as np

from sklearn import tree, ensemble, neighbors, linear_model, neural_network, svm, naive_bayes
from sklearn import model_selection

from model_metrics import ModelMetrics

class Model:
	counter = 0			# number of trained classifiers

	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		self.classifier = classifier
		Model.counter += 1

		self.name_ = 'classifier_' + str(Model.counter)
		self.type_ = ''
		self.comment_ = ''

		if val_method == 'holdout':
			self.train_metrics, self.val_metrics = self.__holdOutValidation(\
				classifier, X, y, val_size, stratify)
		elif val_method == 'cv':
			self.train_metrics, self.val_metrics = self.__crossValidation(\
				classifier, X, y, k, stratify)
		elif val_method == 'loo':
			self.train_metrics, self.val_metrics = self.__crossValidation(\
				classifier, X, y, X.shape[0])

	# hold out a validation set for assessing the performance of classifier
	# 
	# classifier: 	classifier to be assessed
	# X: 			feature values
	# y: 			labels
	# val_size: 	proportion of data reserved for validation
	# stratify: 	whether to statify training and validation data
	#
	# return performance metrics on training and validation data
	def __holdOutValidation(self, classifier, X, y, val_size, stratify=True):
		if stratify:
			s = y
		else:
			s = None

		X_train, X_val, y_train, y_val = model_selection.train_test_split(\
			X, y, test_size=val_size, stratify=s, random_state=0)

		classifier = classifier.fit(X_train, y_train)
		y_train_pred = classifier.predict(X_train)
		y_val_pred = classifier.predict(X_val)
		y_train_prob = classifier.predict_proba(X_train)
		y_val_prob = classifier.predict_proba(X_val)

		return ModelMetrics(classifier, y_train, y_train_pred, y_train_prob, 'holdout'), \
			ModelMetrics(classifier, y_val, y_val_pred, y_val_prob, 'holdout')

	# split data into k folds, train classifier on k-1 folds and assess it on the remaining fold
	# is equivalent to leave-one-out validation when k = number of samples
	# 
	# classifier: 	classifier to be assessed
	# X: 			feature values
	# y: 			labels
	# k: 			number of folds
	# stratify: 	whether to statify training and validation data
	#
	# return performance metrics on training and validation data
	def __crossValidation(self, classifier, X, y, k, stratify=True):
		if k == X.shape[0]:		# leave-one-out
			kf = model_selection.KFold(n_splits=k)
		else:
			if stratify:
				kf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
			else:
				kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=0)

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

			classifier = classifier.fit(X_train, y_train)
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

	# set classifier name
	def setName(self, name):
		self.name_ = name

	# set comment on classifier
	def setComment(self, comment):
		self.comment_ = comment

	# return classifier name
	def name(self):
		return self.name_

	def type(self):
		return self.type_

	# return comment on classifier
	def comment(self):
		return self.comment_

	# return perfermance metrics on training data
	def metrics(self, option):
		if option == 'train':
			return self.train_metrics
		elif option == 'val':
			return self.val_metrics

class DecisionTree(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)
		
		self.type_ = 'decision tree'
		self.criterion = self.classifier.criterion
		self.max_depth = self.classifier.max_depth
		self.min_samples_split = self.classifier.min_samples_split
		self.min_samples_leaf = self.classifier.min_samples_leaf
		self.class_weight = self.classifier.class_weight

class RandomForest(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'random forest'
		self.n_estimators = self.classifier.n_estimators
		self.criterion = self.classifier.criterion
		self.max_features = self.classifier.max_features
		self.max_depth = self.classifier.max_depth
		self.min_samples_split = self.classifier.min_samples_split
		self.min_samples_leaf = self.classifier.min_samples_leaf
		self.bootstrap = self.classifier.bootstrap
		self.class_weight = self.classifier.class_weight

class KNearestNeighbors(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'k-nearest neighbors'
		self.n_neighbors = self.classifier.n_neighbors
		self.weights = self.classifier.weights
		self.metric = self.classifier.metric

class LogisticRegression(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'logistic regression'
		self.penalty = self.classifier.penalty
		self.tol = self.classifier.tol
		self.C = self.classifier.C
		self.fit_intercept = self.classifier.fit_intercept
		self.class_weight = self.classifier.class_weight
		self.max_iter = self.classifier.max_iter
		self.multi_class = self.classifier.multi_class

class NeuralNetwork(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'neural network'
		self.hidden_layer_sizes = self.classifier.hidden_layer_sizes
		self.activation = self.classifier.activation
		self.alpha = self.classifier.alpha
		self.batch_size = self.classifier.batch_size
		self.learning_rate = self.classifier.learning_rate
		self.learning_rate_init = self.classifier.learning_rate_init
		self.max_iter = self.classifier.max_iter
		self.tol = self.classifier.tol
		self.early_stopping = self.classifier.early_stopping

class SVM(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'svm'
		self.C = self.classifier.C
		self.kernel = self.classifier.kernel
		self.degree = self.classifier.degree
		self.gamma = self.classifier.gamma
		self.coef0 = self.classifier.coef0
		self.tol = self.classifier.tol
		self.class_weight = self.classifier.class_weight
		self.max_iter = self.classifier.max_iter

class NaiveBayes(Model):
	def __init__(self, classifier, X, y, val_method, val_size, k, stratify, mode):
		super().__init__(classifier, X, y, val_method, val_size, k, stratify)

		self.type_ = 'naive bayes'
		self.mode = mode

		if mode == 'gaussian':
			self.priors = self.classifier.priors

		elif mode == 'multinomial':
			self.alpha = self.classifier.alpha
			self.fit_prior = self.classifier.fit_prior
			self.class_prior = self.classifier.class_prior