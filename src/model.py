import pandas as pd
import numpy as np
import sklearn

###############################################################################
# A data structure that stores an ML model and its key statistics.
#
##### Fields #####
# type (str): 							classifier type (e.g. 'decision tree')
# name (str): 							classifier name
# comment (str): 						user-supplied comment
# 
# accuracy (float): 					classifier accuracy
# class_precision (np array):  			per class precision
# avg_precision (float): 				unweighted average precision
# class_recall (np array): 				per class recall
# avg_recall (float): 					unweighted average recall
# class_f1 (np array): 					per class f1 score
# avg_f1 (float): 						unweighted average f1 score
#
# roc_plot (plot): 						plot of ROC curve
# class_auroc (np array): 				per class auroc
# avg_auroc (float): 					unweighted average auroc
# pr_plot (plot): 						plot of precision-recall curve
# class_auprc (np array): 				per class auprc
# avg_auprc (float): 					unweighted average auprc
# confusion_matrix (np array): 			confusion matrix
# confusion_plot (plot): 				plot of confusion matrix
###############################################################################
class Model:
	counter = 0 		# number of trained classifiers (used in default name)

	# constructor
	#
	# modelType (str): 				classifier type
	# accuracy (float): 			classifier accuracy
	# class_precision (np array): 	per class precision
	# class_recall (np array): 		per class recall
	# class_f1 (np array): 			per class f1
	# confusion_matrix (np array): 	confusion matrix
	def __init__(self, modelType, accuracy, class_precision, \
		class_recall, class_f1, confusion_matrix):
		Model.counter += 1

		self.name = 'classifier' + str(Model.counter)
		self.comment = ''
		self.type = modelType

		self.accuracy = accuracy
		self.class_precision = class_precision
		self.class_recall = class_recall
		self.class_f1 = class_f1
		self.confusion_matrix = confusion_matrix

		self.avg_precision = np.mean(self.class_precision)
		self.avg_recall = np.mean(self.class_recall)
		self.avg_f1 = np.mean(self.class_f1)

	# set classifier name
	def setName(self, name):
		self.name = name

	# add comment
	def setComment(self, comment):
		self.comment = comment

	# return classifier name
	def getName(self):
		return self.name

	# return user-supplied comment
	def getComment(self):
		return self.comment

	# return classifier type
	def getType(self):
		return self.type

	# return classifier accuracy
	def getAccuracy(self):
		return self.accuracy

	# return per class precision (as a 1d np array)
	def getPerClassPrecision(self):
		return self.class_precision

	# return average precision
	def getAveragePrecision(self):
		return self.avg_precision

	# return per class recall (as a 1d np array)
	def getPerClassRecall(self):
		return self.class_recall

	# return average recall
	def getAverageRecall(self):
		return self.avg_recall

	# return per class f1 scores (as a 1d np array)
	def getPerClassF1(self):
		return self.class_f1

	# return average f1 score
	def getAverageF1(self):
		return self.avg_f1

	# return ROC curve and per class AUROC
	def getROC(self):
		roc_plot = 'todo'
		class_auroc = 'todo'
		return roc_plot, class_auroc

	# return precision-recall curve and per class AUPRC
	def getPR(self):
		pr_plot = 'todo'
		class_auprc = 'todo'
		return pr_plot, class_auprc

	# return plot of confusion matrix
	def getConfusionMatrix(self):
		confusion_matrix = 'todo'
		return confusion_plot
