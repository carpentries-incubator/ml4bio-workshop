import pandas as pd
import numpy as np
from sklearn import model_selection

# A data structure that stores the dataset (either labeled or unlabeled) and 
# its key statistics.
class Data:

	# data: the dataset (a pandas dataframe)
	# labeled: a bool that indicates whether or not the dataset is labeled
	def __init__(self, data, name, labeled):
		self.data = data
		self.name = name
		self.labeled = labeled
		self.num_rows = self.data.shape[0]
		self.num_columns = self.data.shape[1]

	# return the dataset
	# return type: pandas dataframe
	def getData(self):
		return self.data

	# return the name of the dataset
	def getName(self):
		return self.name

	# return the number of samples in the dataset
	def getNumOfSamples(self):
		return self.num_rows

	# return the number of classes in the dataset
	def getNumOfClasses(self):
		return len(set(self.data.iloc[:, self.num_columns - 1]))

	# count the number of samples that belong to each class
	# return a dictionary (key: class name, value: number of samples)
	def getClassCounts(self):
		classes = list(set(self.data.iloc[:, self.num_columns - 1]))
		counts = [0] * len(classes)
		dic = dict(zip(classes, counts))
		for i in range(0, self.num_rows):
			label = self.data.iloc[i, self.num_columns - 1]
			dic[label] += 1
		return dic

	# return the number of features
	def getNumOfFeatures(self):
		if self.labeled:
			return self.num_columns - 1
		else:
			return self.num_columns

	# obtain the type of each feature (numerical or categorical)
	# return a dictionary (key: feature name, value: type)
	def getTypeOfFeatures(self):
		dic = {}
		features = list(self.data)
		for i in range(0, len(features) - 1):
			dic[features[i]] = self.data.dtypes[i]
		return dic

	# split the labeled dataset into training and test sets
	#
	# test_size (float): the proportion of data for testing (e.g. 0.2, 0.33, etc.)
	# stratify (bool): stratified sampling or not
	def trainTestSplit(self, test_size, stratify):
		X = self.data.iloc[:, 0:self.num_columns-1]
		y = self.data.iloc[:, self.num_columns-1]
		if stratify:
			s = y
		else:
			s = None
		X_train, X_test, y_train, y_test = model_selection.train_test_split(\
			X, y, test_size=test_size, stratify=s, random_state=0)
		self.train = [X_train, y_train]
		self.test = [X_test, y_test]



