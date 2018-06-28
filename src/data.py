import pandas as pd

# A data structure that stores the dataset (either labeled or unlabeled) and 
# its key statistics.
class Data:

	# dataframe: a pandas dataframe storing the dataset
	# labeled: a bool that indicates whether or not the dataset is labeled
	def __init__(self, dataframe, name, labeled):
		self.dataframe = dataframe
		self.name = name
		self.labeled = labeled
		self.shape = dataframe.shape

	def getName(self):
		return self.name;

	def getNumOfSamples(self):
		return self.shape[0]

	def getClassCounts(self):
		label_col = self.shape[1] - 1
		classes = list(set(self.dataframe.iloc[:,label_col]))
		counts = [0] * len(classes)
		dic = dict(zip(classes, counts))
		for i in range(0, self.getNumOfSamples()):
			label = self.dataframe.iloc[i, label_col]
			dic[label] += 1
		return dic

	def getNumOfFeatures(self):
		if self.labeled:
			return self.shape[1] - 1
		else:
			return self.shape[1]

	def getTypeOfFeatures(self):
		dic = {}
		features = list(self.dataframe)
		for i in range(0, len(features) - 1):
			dic[features[i]] = self.dataframe.dtypes[i]
		return dic




