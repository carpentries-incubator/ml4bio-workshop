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
# data (pd dataframe):						original dataset
# integer_encoded_data (pd dataframe):		integer encoded dataset
# one_hot_encoded_data (pd dataframe): 		one-hot encoded dataset
# is_encoded (bool):						true if the dataset is encoded
# name (str): 								name of the dataset
# labeled (bool): 							whether or not the dataset is labeled
# num_samples (int): 						number of samples
# num_features (int): 						number of features
# num_one_hot_encoded_features (int):		number of features after one-hot encoding
# num_classes (int): 						number of classes
# features (list): 							collection of features
# one_hot_encoded_features (list): 			collection of features after one-hot encoding
# classes (list): 							collection of class names
# feature_type_dict (dict): 				mapping from feature names to types
# one_hot_encoded_feature_type_dict (dict): mapping from one-hot encoded feature names to types
# class_counts_dict (dict): 				mapping from class names to class counts
# feature_summary (str): 					summary of feature types ('string', 'numeric' or 'mixed')
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
		self.is_encoded = False
		self.num_samples = self.data.shape[0]
		self.num_features = self.data.shape[1]

		# exclude the label column if data is labeled
		if self.labeled:
			self.num_features -= 1

		# extract features and their types
		self.feature_type_dict = dict()
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

		self.__encodeData()
		self.num_one_hot_encoded_features = self.one_hot_encoded_data.shape[1]

		if self.labeled:
			self.num_one_hot_encoded_features -= 1

		self.one_hot_encoded_features = self.one_hot_encoded_data.columns[0: self.num_one_hot_encoded_features]
		self.one_hot_encoded_feature_type_dict = dict(\
			zip(self.one_hot_encoded_features, ['numeric'] * self.num_one_hot_encoded_features))
		self.integer_encoded_feature_type_dict = dict(zip(self.features, ['numeric'] * self.num_features))

		# default train/test split
		self.trainTestSplit(0.2, True)

	# return the original dataset
	def getData(self):
		return self.data

	# return the integer encoded dataset
	def getIntegerEncodedData(self):
		return self.integer_encoded_data

	# return the one-hot encoded dataset
	def getOneHotEncodedData(self):
		return self.one_hot_encoded_data

	# return True if the dataset has been encoded
	def isEncoded(self):
		return self.is_encoded

	# return the name of the dataset
	def getName(self):
		return self.name

	# return the number of samples
	def getNumOfSamples(self):
		return self.num_samples

	# return the number of features
	def getNumOfFeatures(self):
		return self.num_features

	# return the number of features after one-hot encoding
	def getNumOfOneHotEncodedFeatures(self):
		return self.num_one_hot_encoded_features

	# return the list of features
	def getFeatures(self):
		return self.features

	# return the list of one-hot encoded features
	def getOneHotEncodedFeatures(self):
		return self.one_hot_encoded_features

	# return the type of each feature (as a dictionary)
	def getFeatureTypes(self):
		return self.feature_type_dict

	# return the type of each integer encoded feature (as a dictionary)
	def getIntegerEncodedFeatureTypes(self):
		return self.integer_encoded_feature_type_dict

	# return the type of each one-hot encoded feature (as a dictionary)
	def getOneHotEncodedFeatureTypes(self):
		return self.one_hot_encoded_feature_type_dict

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

	# return the mapping from integers to class names (as a dictionary)
	def getClassMap(self):
		return self.class_map

	# return training set
	def getTrain(self):
		return self.train

	# return integer encoded training set
	def getIntegerEncodedTrain(self):
		return self.integer_encoded_train

	# return one-hot encoded training set
	def getOneHotEncodedTrain(self):
		return self.one_hot_encoded_train

	# return test set
	def getTest(self):
		return self.test

	# return integer encoded test set
	def getIntegerEncodedTest(self):
		return self.integer_encoded_test

	# return one-hot encoded test set
	def getOneHotEncodedTest(self):
		return self.one_hot_encoded_test

	# split the labeled dataset into training and test sets
	#
	# test_size (float): 	the proportion of data for testing (e.g. 0.2, 0.33, etc.)
	# stratify (bool): 		stratified sampling or not
	def trainTestSplit(self, test_size, stratify):
		X = self.data.iloc[:, 0: self.num_features]
		y = self.data.iloc[:, self.num_features]
		integer_encoded_X = self.integer_encoded_data.iloc[:, 0: self.num_features]
		integer_encoded_y = self.integer_encoded_data.iloc[:, self.num_features]
		one_hot_encoded_X = self.one_hot_encoded_data.iloc[:, 0: self.num_one_hot_encoded_features]
		one_hot_encoded_y = self.one_hot_encoded_data.iloc[:, self.num_one_hot_encoded_features]
		
		if stratify:
			s = y
		else:
			s = None
		
		X_train, X_test, y_train, y_test = model_selection.train_test_split(\
			X, y, test_size=test_size, stratify=s, random_state=0)
		integer_encoded_X_train, integer_encoded_X_test, integer_encoded_y_train, integer_encoded_y_test = \
			model_selection.train_test_split(integer_encoded_X, integer_encoded_y, test_size=test_size, \
			stratify=s, random_state=0)
		one_hot_encoded_X_train, one_hot_encoded_X_test, one_hot_encoded_y_train, one_hot_encoded_y_test = \
			model_selection.train_test_split(one_hot_encoded_X, one_hot_encoded_y, test_size=test_size, \
			stratify=s, random_state=0)
		self.train = pd.concat([X_train, y_train], axis=1)
		self.test = pd.concat([X_test, y_test], axis=1)
		self.integer_encoded_train = pd.concat([integer_encoded_X_train, integer_encoded_y_train], axis=1)
		self.integer_encoded_test = pd.concat([integer_encoded_X_test, integer_encoded_y_test], axis=1)
		self.one_hot_encoded_train = pd.concat([one_hot_encoded_X_train, one_hot_encoded_y_train], axis=1)
		self.one_hot_encoded_test = pd.concat([one_hot_encoded_X_test, one_hot_encoded_y_test], axis=1)

	# encode discrete features
	def __encodeData(self):
		self.integer_encoded_data = pd.DataFrame([])
		self.one_hot_encoded_data = pd.DataFrame([])
		self.class_map = dict()
		le = preprocessing.LabelEncoder()		# integer encoder
		ohe = preprocessing.OneHotEncoder()		# one-hot encoder

		for i in range(0, self.num_features):
			if pd.api.types.is_string_dtype(self.data.iloc[:, i]):
				feature_name = self.data.columns[i]

				le_col = pd.DataFrame(le.fit_transform(self.data.iloc[:, i]))
				le_col.columns = [feature_name]
				num_items = len(le_col[feature_name].unique())	# number of values of a string-valued feature
				self.integer_encoded_data = pd.concat([self.integer_encoded_data, le_col], axis=1)

				ohe_cols = pd.DataFrame(ohe.fit_transform(le_col).toarray())
				item_names = list(le.inverse_transform(list(range(0, num_items))))	# a collection of values
				col_names = [feature_name + '_is_' + s for s in item_names]	# construct descriptive column names
				ohe_cols.columns = col_names
				self.one_hot_encoded_data = pd.concat([self.one_hot_encoded_data, ohe_cols], axis=1)

				self.is_encoded = True
			else:
				self.integer_encoded_data = pd.concat([self.integer_encoded_data, self.data.iloc[:,i]], axis=1)
				self.one_hot_encoded_data = pd.concat([self.one_hot_encoded_data, self.data.iloc[:,i]], axis=1)

		# add back the label column
		label_col = pd.DataFrame(le.fit_transform(self.data.iloc[:, self.num_features]))
		num_classes = self.getNumOfClasses()
		class_names = list(le.inverse_transform(list(range(0, num_classes))))
		
		for i in range(0, num_classes):
			self.class_map[i] = class_names[i]

		self.integer_encoded_data = pd.concat([self.integer_encoded_data, label_col], axis=1)
		self.one_hot_encoded_data = pd.concat([self.one_hot_encoded_data, label_col], axis=1)