import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

###############################################################################
# A data structure that stores a dataset and its key statistics.
# 
##### Assumptions #####
# a single classification task (i.e. one label column)
# the labels are in the last column
# categorical labels
#
##### Fields #####
# data (pd dataframe):					original dataset
# transformed_data (pd dataframe):		transformed dataset (one-hot encoding)
# is_transformed (bool):				true if the dataset is transformed
# name (str): 							name of the dataset
# labeled (bool): 						whether or not the dataset is labeled
# num_samples (int): 					number of samples
# num_features (int): 					number of features
# num_transformedFeatures (int):		number of features after transformation
# num_classes (int): 					number of classes
# features (list): 						collection of features
# transformed_features (list): 			collection of transformed features
# classes (list): 						collection of class names
# feature_type_dict (dict): 			mapping from feature names to types
# transformed_feature_type_dict (dict): mapping from transformed feature names to types
# class_counts_dict (dict): 			mapping from class names to class counts
# feature_summary (str): 				summary of feature types ('string', 'numeric' or 'mixed')
###############################################################################
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
		self.is_transformed = False
		self.num_samples = self.data.shape[0]
		self.num_features = self.data.shape[1]

		# exclude the label column if data is labeled
		if self.labeled:
			self.num_features -= 1

		# extract features and their types
		self.feature_type_dict = {}
		num_string = 0
		num_numeric = 0
		self.features = self.data.columns[0: self.num_features]

		for i in range(0, self.num_features):
			if pd.api.types.is_string_dtype(self.data.iloc[:, i]):
				self.feature_type_dict[self.features[i]] = 'string'
				num_string += 1
			else:
				self.feature_type_dict[self.features[i]] = 'numeric'
				num_numeric += 1

		if num_string > 0 and num_numeric == 0:
			self.feature_summary = 'string'
		elif num_string == 0 and num_numeric > 0:
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

		self.__transformData()
		self.num_transformed_features = self.transformed_data.shape[1]

		if self.labeled:
			self.num_transformed_features -= 1

		self.transformed_features = self.transformed_data.columns[0: self.num_transformed_features]
		self.transformed_feature_type_dict = dict(\
			zip(self.transformed_features, ['numeric'] * self.num_transformed_features))

		# default train/test split
		self.trainTestSplit(0.2, True)

	# return the original dataset
	def getData(self):
		return self.data

	# return the transformed dataset
	def getTransformedData(self):
		return self.transformed_data

	# return true if the data is transformed
	def isTransformed(self):
		return self.is_transformed

	# return the name of the dataset
	def getName(self):
		return self.name

	# return the number of samples
	def getNumOfSamples(self):
		return self.num_samples

	# return the number of features
	def getNumOfFeatures(self):
		return self.num_features

	# return the number of features after transformation
	def getNumOfTransformedFeatures(self):
		return self.num_transformed_features

	# return the list of features
	def getFeatures(self):
		return self.features

	# return the list of transformed features
	def getTransformedFeatures(self):
		return self.transformed_features

	# return the type of each feature (as a dictionary)
	def getFeatureTypes(self):
		return self.feature_type_dict

	# return the type of each transformed feature (as a dictionary)
	def getTransformedFeatureTypes(self):
		return self.transformed_feature_type_dict

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

	# return training set
	def getTrain(self):
		return self.train

	# return transformed training set
	def getTransformedTrain(self):
		return self.transformed_train

	# return test set
	def getTest(self):
		return self.test

	# return transformed test set
	def getTransformedTest(self):
		return self.transformed_test

	# split the labeled dataset into training and test sets
	# use transformed data
	#
	# test_size (float): 	the proportion of data for testing (e.g. 0.2, 0.33, etc.)
	# stratify (bool): 		stratified sampling or not
	def trainTestSplit(self, test_size, stratify):
		X = self.data.iloc[:, 0: self.num_features]
		y = self.data.iloc[:, self.num_features]
		transformed_X = self.transformed_data.iloc[:, 0: self.num_transformed_features]
		transformed_y = self.transformed_data.iloc[:, self.num_transformed_features]
		
		if stratify:
			s = y
		else:
			s = None
		
		X_train, X_test, y_train, y_test = model_selection.train_test_split(\
			X, y, test_size=test_size, stratify=s, random_state=0)
		transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test = \
			model_selection.train_test_split(transformed_X, transformed_y, test_size=test_size,\
			stratify=s, random_state=0)
		self.train = pd.concat([X_train, y_train], axis=1)
		self.test = pd.concat([X_test, y_test], axis=1)
		self.transformed_train = pd.concat([transformed_X_train, transformed_y_train], axis=1)
		self.transformed_test = pd.concat([transformed_X_test, transformed_y_test], axis=1)

	# transform string-valued features using one-hot encoding
	def __transformData(self):
		self.transformed_data = pd.DataFrame([])
		le = preprocessing.LabelEncoder()		# label encoder (encode strings by integers)
		ohe = preprocessing.OneHotEncoder()		# one-hot encoder

		for i in range(0, self.num_features):
			if pd.api.types.is_string_dtype(self.data.iloc[:, i]):
				feature_name = self.data.columns[i]
				le_col = le.fit_transform(self.data.iloc[:, i])
				num_items = len(set(le_col))	# number of values of a string-valued feature
				item_names = list(le.inverse_transform(list(range(0, num_items))))	# a collection of values
				col_names = [feature_name + '_is_' + s for s in item_names]	# construct descriptive column names
				ohe_cols = pd.DataFrame(ohe.fit_transform(pd.DataFrame(le_col)).toarray())
				ohe_cols.columns = col_names
				self.transformed_data = pd.concat([self.transformed_data, ohe_cols], axis=1)
				self.is_transformed = True
			else:
				self.transformed_data = pd.concat([self.transformed_data, self.data.iloc[:,i]], axis=1)

		# add back the label column
		self.transformed_data = pd.concat([self.transformed_data, self.data.iloc[:, self.num_features]], axis=1)