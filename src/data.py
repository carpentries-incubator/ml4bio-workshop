import pandas as pd
import numpy as np
from sklearn import model_selection

# A data structure that stores a dataset and its key statistics.
# 
# Assumptions
# - a single classification task
# - the labels are in the last column
# - categorical labels
#
# Fields
# data (pd dataframe):		dataset
# name (str): 				name of the dataset
# labeled (bool): 			whether or not the dataset is labeled
# num_samples (int): 		number of samples
# num_features (int): 		number of features
# num_classes (int): 		number of classes
# features (list): 			collection of feature names
# classes (list): 			collection of class names
# feature_type_dict (dict): mapping from feature names to feature types
# class_counts_dict (dict): mapping from class names to class counts
# feature_summary: 			summary of feature types ('string', 'numeric' or 'mixed')
# train: 					training set
# test: 					test set
class Data:

	# constructor
	# 
	# data (pd dataframe): 	dataset
	# name (str): 			name of the dataset
	# labeled (bool): 		whether or not the dataset is labeled
	def __init__(self, data, name, labeled):
		self.data = data
		self.name = name
		self.labeled = labeled
		self.num_samples = self.data.shape[0]
		
		# count the number of features
		if self.labeled:
			self.num_features = self.data.shape[1] - 1
		else:
			self.num_features = self.data.shape[1]

		# extract features and their types
		self.feature_type_dict = {}
		string = 0
		numeric = 0
		col_names = list(self.data)
		self.features = col_names[0: self.num_features]

		for i in range(0, self.num_features):
			if pd.api.types.is_string_dtype(self.data.iloc[:, i]):
				self.feature_type_dict[self.features[i]] = 'string'
				string += 1
			else:
				self.feature_type_dict[self.features[i]] = 'numeric'
				numeric += 1

		if string > 0 and numeric == 0:
			self.feature_summary = 'string'
		elif string == 0 and numeric > 0:
			self.feature_summary = 'numeric'
		else:
			self.feature_summary = 'mixed'

		# extract classes and their counts
		self.classes = list(set(self.data.iloc[:, self.num_features]))
		self.num_classes = len(self.classes)
		class_counts = [0] * self.num_classes
		self.class_counts_dict = dict(zip(self.classes, class_counts))

		for i in range(0, self.num_samples):
			label = self.data.iloc[i, self.num_features]
			self.class_counts_dict[label] += 1

	# return the dataset
	def getData(self):
		return self.data

	# return the name of the dataset
	def getName(self):
		return self.name

	# return the number of samples
	def getNumOfSamples(self):
		return self.num_samples

	# return the number of features
	def getNumOfFeatures(self):
		return self.num_features

	# return the list of features
	def getFeatures(self):
		return self.features

	# return the type of each feature (as a dictionary)
	def getFeatureTypes(self):
		return self.feature_type_dict

	# return the feature summary ('string', 'numeric' or 'mixed')
	def getFeatureSummary(self):
		return self.feature_summary

	# return the number of classes
	def getNumOfClasses(self):
		return self.num_classes

	# return the list of classes
	def getClasses(self):
		return self.classes

	# return the count of each class (as a dictionary)
	def getClassCounts(self):
		return self.class_counts_dict

	# split the labeled dataset into training and test sets
	#
	# test_size (float): 	the proportion of data for testing (e.g. 0.2, 0.33, etc.)
	# stratify (bool): 		stratified sampling or not
	def trainTestSplit(self, test_size, stratify):
		X = self.data.iloc[:, 0: self.num_features]
		y = self.data.iloc[:, self.num_features]
		
		if stratify:
			s = y
		else:
			s = None
		
		X_train, X_test, y_train, y_test = model_selection.train_test_split(\
			X, y, test_size=test_size, stratify=s, random_state=0)
		self.train = [X_train, y_train]
		self.test = [X_test, y_test]

		return self.train, self.test



