import os
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from PyQt5.QtWidgets import QTreeWidgetItem

class Data:
	"""
	An instance of this class stores the data used for learning and prediction.
	The stored information includes:

		- labeled data
		- unlabeled data (if uploaded)
		- labeled file name
		- unlabeled file name (if uploaded)
		- number of samples in labeled data
		- number of samples in unlabeled data (if uploaded)
		- number of features
		- type of features (individual and overall)
		- number of valid classes
		- number of samples per class
		- encoded data (integer and one-hot encoding)

	:ivar labeled_data: labeled data for learning classifiers
	:vartype labeled_data: pandas dataframe
	:ivar unlabeled_data: unlabeled data for prediction
	:vartype unlabeled_data: pandas dataframe
	:ivar labeled_name: name of the labeled dataset
	:vartype labeled_name: str
	:ivar unlabeled_name: name of the unlabeled dataset
	:vartype unlabeled_name: str
	:ivar labeled_num_samples: number of samples in the labeled dataset
	:vartype labeled_num_samples: int
	:ivar unlabeled_num_samples: number of samples in the unlabeled dataset
	:vartype unlabeled_num_samples: int
	:ivar num_features\_: number of features
	:vartype num_features\_: int
	:ivar individual_feature_type: type of each feature in a dictionary whose keys are feature names
	:vartype individual_feature_type: dict
	:ivar global_feature_type: an overall summary of feature types ('continuous', 'discrete', 'mixed')
	:vartype global_feature_type: str
	:ivar num_classes\_: number of valid classes
	:vartype num_classes\_: int
	:ivar class_num_samples: number of samples per class in a dictionary whose keys are class names
	:vartype class_num_samples: dict
	:ivar integer_encoded_labeled_data: integer-encoded labeled data
	:vartype integer_encoded_labeled_data: pandas dataframe
	:ivar integer_encoded_unlabeled_data: integer-encoded unlabeled data
	:vartype integer_encoded_unlabeled_data: pandas dataframe
	:ivar one_hot_encoded_labeled_data: one-hot-encoded labeled data
	:vartype one_hot_encoded_labeled_data: pandas dataframe
	:ivar one_hot_encoded_unlabeled_data: one-hot-encoded unlabeled data
	:vartype one_hot_encoded_unlabeled_data: pandas dataframe
	:ivar num_one_hot_encoded_features: number of features after one-hot encoding
	:vartype num_one_hot_encoded_features: int
	"""

	def __init__(self, path):
		"""
		Constructs a new instance of the class.

		:param path: file path of the labeled data
		:type path: str
		"""
		self.labeled_data = pd.read_csv(path, sep=None, engine='python')
		self.unlabeled_data = None 		# unlabeled data is optional
		self.labeled_name = os.path.basename(path)
		self.labeled_num_samples = self.labeled_data.shape[0]

		self.num_features_ = self.labeled_data.shape[1] - 1
		feature_names = self.labeled_data.columns[0: self.num_features_]
		num_continuous_features = 0
		num_discrete_features = 0
		
		# determine feature types
		self.individual_feature_type = dict()
		for f in feature_names:
			dtype = self.labeled_data.dtypes[f]
			if pd.api.types.is_string_dtype(dtype):
				num_discrete_features += 1
				self.individual_feature_type[f] = 'discrete'
			elif pd.api.types.is_integer_dtype(dtype):
				if len(set(self.labeled_data[f])) <= 10:
					num_discrete_features += 1
					self.individual_feature_type[f] = 'discrete'
				else:
					num_continuous_features += 1
					self.individual_feature_type[f] = 'continuous'
			else:
				num_continuous_features += 1
				self.individual_feature_type[f] = 'continuous'

		# determine global feature type
		if num_continuous_features > 0 and num_discrete_features == 0:
			self.global_feature_type = 'continuous'
		elif num_continuous_features == 0 and num_discrete_features > 0:
			self.global_feature_type = 'discrete'
		else:
			self.global_feature_type = 'mixed'

		label_col = self.labeled_data.iloc[:, self.num_features_]
		classes = list(set(label_col))
		self.num_classes_ = len(classes)
		
		# count number of samples that belong to each class
		self.class_num_samples = dict()
		for c in classes:
			self.class_num_samples[c] = 0
		for l in label_col:
			self.class_num_samples[l] += 1

	def add_unlabeled_data(self, path):
		"""
		Adds unlabeled data for prediction.

		:param path: file path of the unlabeled data
		:type path: str
		"""
		try:
			self.unlabeled_data = pd.read_csv(path, sep=None, engine='python')
		except:
			raise
		
		labeled_feature_names = list(self.labeled_data.columns[0: self.num_features()])
		unlabeled_feature_names = list(self.unlabeled_data.columns)
		if labeled_feature_names != unlabeled_feature_names:
			self.unlabeled_data = None
			raise ValueError()
		
		self.unlabeled_name = os.path.basename(path)
		self.unlabeled_num_samples = self.unlabeled_data.shape[0]

	def encode(self):
		"""
		Encodes the data using integer and one-hot encoders.
		"""
		raw_data = self.labeled_data.iloc[:, 0: self.num_features()]
		if self.unlabeled_data is not None:
			raw_data = pd.concat([raw_data, self.unlabeled_data], axis=0)

		integer_encoded_data = pd.DataFrame([])
		one_hot_encoded_data = pd.DataFrame([])
		le = preprocessing.LabelEncoder()		# integer encoder
		ohe = preprocessing.OneHotEncoder()		# one-hot encoder

		feature_type = self.feature_type(option='individual')
		for i in range(0, self.num_features()):
			feature = raw_data.columns[i]
			if feature_type[feature] == 'discrete':
				le_col = le.fit_transform(raw_data[feature])
				num_values = len(np.unique(le_col))	# number of values of a discrete feature
				le_col = pd.DataFrame(le_col)
				le_col.columns = [feature]
				integer_encoded_data = pd.concat([integer_encoded_data, le_col], axis=1)

				ohe_cols = pd.DataFrame(ohe.fit_transform(pd.DataFrame(le_col)).toarray())
				values = list(le.inverse_transform(list(range(0, num_values))))	# a collection of values
				col_names = [feature + '=' + str(v) for v in values]	# construct descriptive column names
				ohe_cols.columns = col_names
				one_hot_encoded_data = pd.concat([one_hot_encoded_data, ohe_cols], axis=1)
			else:
				integer_encoded_data = pd.concat([integer_encoded_data, raw_data[feature]], axis=1)
				one_hot_encoded_data = pd.concat([one_hot_encoded_data, raw_data[feature]], axis=1)

		self.integer_encoded_labeled_data = integer_encoded_data[0: self.num_samples('labeled')] 	# no label column
		self.one_hot_encoded_labeled_data = one_hot_encoded_data[0: self.num_samples('labeled')]	# no label column
		self.num_one_hot_encoded_features = len(self.one_hot_encoded_labeled_data.columns)

		if self.unlabeled_data is not None:
			self.integer_encoded_unlabeled_data = integer_encoded_data[self.num_samples('labeled'): ]
			self.one_hot_encoded_unlabeled_data = one_hot_encoded_data[self.num_samples('labeled'): ]

	def split(self, test_size, stratify):
		"""
		Splits labeled data into training and test sets.

		:param test_size: percent of test data
		:type test_size: float
		:param stratify: draw samples according to class proportions or not
		:type stratify: bool
		"""
		integer_encoded_X = self.integer_encoded_labeled_data
		one_hot_encoded_X = self.one_hot_encoded_labeled_data
		y = self.labeled_data.iloc[:, self.num_features()]
		
		if stratify:
			s = y
		else:
			s = None

		integer_encoded_X_train, integer_encoded_X_test, integer_encoded_y_train, integer_encoded_y_test = \
			model_selection.train_test_split(integer_encoded_X, y, test_size=test_size, stratify=s, random_state=0)
		one_hot_encoded_X_train, one_hot_encoded_X_test, one_hot_encoded_y_train, one_hot_encoded_y_test = \
			model_selection.train_test_split(one_hot_encoded_X, y, test_size=test_size, stratify=s, random_state=0)

		self.integer_encoded_train = pd.concat([integer_encoded_X_train, integer_encoded_y_train], axis=1)
		self.integer_encoded_test = pd.concat([integer_encoded_X_test, integer_encoded_y_test], axis=1)
		self.one_hot_encoded_train = pd.concat([one_hot_encoded_X_train, one_hot_encoded_y_train], axis=1)
		self.one_hot_encoded_test = pd.concat([one_hot_encoded_X_test, one_hot_encoded_y_test], axis=1)

	def name(self, option='labeled'):
		"""
		Returns name of the dataset.

			- option='labeled': name of labeled data
			- option='unlabeled': name of unlabeled data

		:param option: 'labeled' or 'unlabeled'
		:type option: str

		:returns: name of the dataset
		"""
		if option == 'labeled':		return self.labeled_name
		elif option == 'unlabeled':	return self.unlabeled_name

	def num_samples(self, option='labeled'):
		"""
		Returns number of samples.

			- option='labeled': total number of samples in labeled data
			- option='unlabeled': total number of samples in unlabeled data
			- option='classwise': number of samples that belong to each class

		:param option: 'labeled', 'unlabeled' or 'classwise'
		:type option: str

		:returns: number of samples
		"""
		if option == 'labeled':		return self.labeled_num_samples
		elif option == 'unlabeled':	return self.unlabeled_num_samples
		elif option == 'classwise':	return self.class_num_samples

	def num_features(self, option='raw'):
		"""
		Returns number of features.

			- option='raw': number of raw features (raw and integer-encoded data)
			- option='one-hot': number of features after one-hot encoding

		:param option: 'raw','integer' or 'one-hot'
		:type option: str

		:returns: number of features
		"""
		if option in ['raw', 'integer']:	return self.num_features_
		elif option == 'one-hot':			return self.num_one_hot_encoded_features

	def feature_type(self, option='global'):
		"""
		Returns summary of feature types.

			- option='individual': type of each feature
			- option='global': overall summary of feature types

		:param option: 'individual' or 'global'
		:type option: str

		:returns: summary of feature types
		"""
		if option == 'individual':	return self.individual_feature_type
		elif option == 'global':	return self.global_feature_type

	def num_classes(self):
		"""
		Returns number of valid classes.

		:returns: number of valid classes
		"""
		return self.num_classes_

	def summary(self, view):
		"""
		Displays a summary of the loaded data. This includes 

			- number of samples (total and classwise)
			- number of features
			- feature types (individual and global)

		The summary is organized in a tree structure.

		:param view: the widget where the summary is displayed
		:type view: QTreeWidget
		"""
		view.clear()

		samples_item = QTreeWidgetItem(view)
		samples_item.setText(0, 'Samples')
		labeled_samples_item = QTreeWidgetItem(samples_item)
		labeled_samples_item.setText(0, 'labeled')
		labeled_samples_item.setText(1, str(self.num_samples()))
		
		# show number of samples that belong to each class
		class_num_samples = self.num_samples('classwise')
		for c in class_num_samples:
			class_samples_item = QTreeWidgetItem(labeled_samples_item)
			class_samples_item.setText(0, str(c))
			class_samples_item.setText(1, str(class_num_samples[c]))
			class_samples_item.setToolTip(0, str(c))
		
		if self.unlabeled_data is not None:
			unlabeled_samples_item = QTreeWidgetItem(samples_item)
			unlabeled_samples_item.setText(0, 'unlabeled')
			unlabeled_samples_item.setText(1, str(self.num_samples('unlabeled')))

		features_item = QTreeWidgetItem(view)
		features_item.setText(0, 'Features')
		features_item.setText(1, str(self.num_features()))
		
		# show type of each feature
		feature_type = self.feature_type(option='individual')
		for f in feature_type:
			feature_type_item = QTreeWidgetItem(features_item)
			feature_type_item.setText(0, f)
			feature_type_item.setText(1, feature_type[f])
			feature_type_item.setToolTip(0, f)

	def train(self, option):
		"""
		Returns encoded training data.

			- option='integer': integer-encoded data
			- option='one-hot': one-hot-encoded data

		:param option: 'integer' or 'one-hot'
		:type option: str

		:returns: encoded training data
		"""
		if option == 'integer':		return self.integer_encoded_train
		elif option == 'one-hot':	return self.one_hot_encoded_train

	def test(self, option):
		"""
		Returns encoded test data.

			- option='integer': integer-encoded data
			- option='one-hot': one-hot-encoded data	

		:param option: 'integer' or 'one-hot'
		:type option: str

		:returns: encoded test data
		"""
		if option == 'integer':		return self.integer_encoded_test
		elif option == 'one-hot':	return self.one_hot_encoded_test

	def prediction(self, option):
		"""
		Returns encoded unlabeled data for prediction.

			- option='integer': integer-encoded data
			- option='one-hot': one-hot-encoded data

		:param option: 'integer' or 'one-hot'
		:type option: str

		:returns: encoded unlabeled data
		"""
		if option == 'integer':		return self.integer_encoded_unlabeled_data
		elif option == 'one-hot':	return self.one_hot_encoded_unlabeled_data