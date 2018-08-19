import numpy as np
import pandas as pd
from itertools import cycle
from sklearn import metrics
from PyQt5.QtWidgets import QTreeWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import rcParams


class ModelMetrics:
	"""
	An instance of this class stores information about a classifier's 
	performance metrics with respect to training, validation or test set.
	Stored metrics include:

		- accuracy
		- precision (average and classwise)
		- recall (average and classwise)
		- f1 score (average and classwise)
		- AUROC (average and classwise)
		- AUPRC (average and classwise)
		- confusion matrix

	:ivar classes\_: valid classes for the classification problem
	:vartype classes\_: list
	:ivar num_classes\_: total number of valid classes
	:vartype num_classes\_: int
	:ivar accuracy\_: accuracy of the classifier
	:vartype accuracy\_: float
	:ivar avg_precision\_: average precision over all classes
	:vartype avg_precision\_: float
	:ivar avg_recall\_: average recall over all classes
	:vartype avg_recall\_: float
	:ivar avg_f1\_: average f1 score over all classes
	:vartype avg_f1\_: float
	:ivar avg_auroc\_: average AUROC over all classes
	:vartype avg_auroc\_: float
	:ivar avg_auprc\_: average AUPRC over all classes
	:vartype avg_auprc\_: float
	:ivar confusion_matrix\_: confusion matrix in percents
	:vartype confusion_matrix\_: 2D array

	The dictionaries below use class names as keys.
	Each value in the dictionaries is a floating-point number.

	:ivar class_precision\_: classwise precisions
	:vartype class_precision\_: dict
	:ivar class_recall\_: classwise recalls
	:vartype class_recall\_: dict
	:ivar class_f1\_: classwise f1 scores
	:vartype class_f1\_: dict
	:ivar class_auroc\_: classwise AUROCs
	:vartype class_auroc\_: dict
	:ivar class_auprc\_: classwise AUPRCs
	:vartype class_auprc\_: dict

	The dictionaries below use class names and 'avg' as keys.
	Each value in the dictionaries is a list of metric values at different decision threshold.

	:ivar fpr\_: FPRs for plotting all ROC curves
	:vartype fpr\_: dict
	:ivar tpr\_: TPRs for plotting all ROC curves
	:vartype tpr\_: dict
	:ivar precision\_: precisions for plotting all PR curves
	:vartype precision\_: dict
	:ivar recall\_: recalls for plotting all PR curves
	:vartype recall\_: dict
	"""

	def __init__(self, classifier, y, y_pred, y_prob, option):
		"""
		Constructs a new instance of the class.

		:param classifier: classifier for evaluation
		:type classifier: sklearn classifier object
		:param y: true labels
		:type y: pandas series
		:param y_pred: predicted labels
		:type y_pred: pandas series
		:param y_prob: classwise probabilities
		:type y_prob: pandas dataframe
		:param option: specification of evaluation method ("holdout", "cv" or "loo")
		:type option: str
		"""
		classes = classifier.classes_
		num_classes = len(classes)
		self.classes_ = classes
		self.num_classes_ = num_classes

		fpr = dict()
		tpr = dict()
		auroc = dict()

		precision = dict()
		recall = dict()
		auprc = dict()

		if option in ['holdout', 'loo']:
			self.accuracy_ = np.around(metrics.accuracy_score(y, y_pred), decimals=3)
			class_precision = metrics.precision_score(y, y_pred, average=None)
			class_recall = metrics.recall_score(y, y_pred, average=None)
			class_f1 = metrics.f1_score(y, y_pred, average=None)
			confusion_matrix = metrics.confusion_matrix(y, y_pred)

			for i in range(0, num_classes):
				fpr[i], tpr[i], _ = metrics.roc_curve(y, y_prob[:, i], pos_label=classes[i])
				auroc[classes[i]] = np.around(metrics.auc(fpr[i], tpr[i]), decimals=3)

				precision[i], recall[i], _ = metrics.precision_recall_curve(y, y_prob[:, i], pos_label=classes[i])
				precision[i] = np.flip(precision[i], 0)
				recall[i] = np.flip(recall[i], 0)
				auprc[classes[i]] = np.around(metrics.auc(recall[i], precision[i]), decimals=3)

		elif option == 'cv':
			k = len(y)
			accuracy = 0
			class_precision = np.zeros(num_classes)
			class_recall = np.zeros(num_classes)
			class_f1 = np.zeros(num_classes)
			confusion_matrix = np.zeros([num_classes, num_classes])

			# classwise metrics w.r.t. each fold
			for y_i, y_pred_i in zip(y, y_pred):
				accuracy += metrics.accuracy_score(y_i, y_pred_i)
				class_precision += metrics.precision_score(y_i, y_pred_i, average=None)
				class_recall += metrics.recall_score(y_i, y_pred_i, average=None)
				class_f1 += metrics.f1_score(y_i, y_pred_i, average=None)
				confusion_matrix += metrics.confusion_matrix(y_i, y_pred_i)

			# classwise metrics average over all folds
			self.accuracy_ = np.around(accuracy / k, decimals=3)
			class_precision = class_precision / k
			class_recall = class_recall / k
			class_f1 = class_f1 / k

			for i in range(0, num_classes):
				fpr_i = dict()
				tpr_i = dict()
				precision_i = dict()
				recall_i = dict()

				for j in range(0, k):
					fpr_i[j], tpr_i[j], _ = metrics.roc_curve(y[j], y_prob[j][:, i], pos_label=classes[i])
					precision_i[j], recall_i[j], _ = metrics.precision_recall_curve(y[j], y_prob[j][:, i], pos_label=classes[i])
				
				fpr[i] = np.unique(np.concatenate([fpr_i[j] for j in range(0, k)]))
				tpr[i] = np.zeros_like(fpr[i])
				recall[i] = np.unique(np.concatenate([recall_i[j] for j in range(0, k)])) 	# all unique x-values
				precision[i] = np.zeros_like(recall[i])

				# interpolate y-values
				for j in range(0, k):
					tpr[i] += np.interp(fpr[i], fpr_i[j], tpr_i[j])
					precision[i] += np.interp(recall[i], np.flip(recall_i[j], 0), np.flip(precision_i[j], 0))
				
				tpr[i] /= k
				precision[i] /= k
				auroc[classes[i]] = np.around(metrics.auc(fpr[i], tpr[i]), decimals=3)
				auprc[classes[i]] = np.around(metrics.auc(recall[i], precision[i]), decimals=3)

		# metrics averaged over all classes
		self.avg_precision_ = np.around(np.mean(class_precision), decimals=3)
		self.avg_recall_ = np.around(np.mean(class_recall), decimals=3)
		self.avg_f1_ = np.around(np.mean(class_f1), decimals=3)

		self.class_precision_ = dict(zip(classes, np.around(class_precision, decimals=3)))
		self.class_recall_ = dict(zip(classes, np.around(class_recall, decimals=3)))
		self.class_f1_ = dict(zip(classes, np.around(class_f1, decimals=3)))

		#confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
		self.confusion_matrix_ = np.around(confusion_matrix, decimals=3)
		
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0, num_classes)]))
		all_recall = np.unique(np.concatenate([recall[i] for i in range(0, num_classes)]))
		mean_tpr = np.zeros_like(all_fpr)
		mean_precision = np.zeros_like(all_recall)

		for i in range(0, num_classes):
			mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
			mean_precision += np.interp(all_recall, recall[i], precision[i])

		mean_tpr /= num_classes
		mean_precision /= num_classes

		fpr['avg'] = all_fpr
		tpr['avg'] = mean_tpr
		self.avg_auroc_ = np.around(metrics.auc(fpr['avg'], tpr['avg']), decimals=3)

		precision['avg'] = mean_precision
		recall['avg'] = all_recall
		self.avg_auprc_ = np.around(metrics.auc(recall['avg'], precision['avg']), decimals=3)

		self.fpr_ = fpr
		self.tpr_ = tpr
		self.class_auroc_ = auroc

		self.precision_ = precision
		self.recall_ = recall
		self.class_auprc_ = auprc

	def accuracy(self):
		"""
		Returns accuracy of the classifier.
		"""
		return self.accuracy_

	def precision(self, option='average'):
		"""
		Returns precision of the classifier.

			- option='average': average precision
			- option='classwise': classwise precision

		:param option: 'average' or 'classwise'
		:type option: str
		"""
		if option == 'classwise':	return self.class_precision_
		elif option == 'average':	return self.avg_precision_

	def recall(self, option='average'):
		"""
		Returns recall of the classifier.

			- option='average': average recall
			- option='classwise': classwise recall

		:param option: 'average' or 'classwise'
		:type option: str
		"""
		if option == 'classwise':	return self.class_recall_
		elif option == 'average':	return self.avg_recall_

	def f1(self, option='average'):
		"""
		Returns f1 score of the classifier.

			- option='average': average f1 score
			- option='classwise': classwise f1 score

		:param option: 'average' or 'classwise'
		:type option: str
		"""
		if option == 'classwise':	return self.class_f1_
		elif option == 'average':	return self.avg_f1_

	def auroc(self, option='average'):
		"""
		Returns AUROC of the classifier.

			- option='average': average AUROC
			- option='classwise': classwise AUROC

		:param option: 'average' or 'classwise'
		:type option: str
		"""
		if option == 'classwise':	return self.class_auroc_
		elif option == 'average':	return self.avg_auroc_

	def auprc(self, option='average'):
		"""
		Returns AUPRC of the classifier.

			- option='average': average AUPRC
			- option='classwise': classwise AUPRC

		:param option: 'average' or 'classwise'
		:type option: str
		"""
		if option == 'classwise':	return self.class_auprc_
		elif option == 'average':	return self.avg_auprc_

	def summary(self, parent):
		"""
		Displays a summary of performance metrics.
		The summary is organized in a tree structure.

		:param parent: the parent item in the model summary
		:type parent: QTreeWidgetItem
		"""
		accuracy_item = QTreeWidgetItem(parent)
		accuracy_item.setText(0, 'Accuracy')
		accuracy_item.setText(1, str(self.accuracy()))

		precision_item = QTreeWidgetItem(parent)
		precision_item.setText(0, 'Precision')
		precision_item.setText(1, str(self.precision()))
		class_precision = self.precision('classwise')
		self.class_summary(precision_item, class_precision)

		recall_item = QTreeWidgetItem(parent)
		recall_item.setText(0, 'Recall')
		recall_item.setText(1, str(self.recall()))
		class_recall = self.recall('classwise')
		self.class_summary(recall_item, class_recall)

		f1_item = QTreeWidgetItem(parent)
		f1_item.setText(0, 'F1')
		f1_item.setText(1, str(self.f1()))
		class_f1 = self.f1('classwise')
		self.class_summary(f1_item, class_f1)

		auroc_item = QTreeWidgetItem(parent)
		auroc_item.setText(0, 'AUROC')
		auroc_item.setText(1, str(self.auroc()))
		class_auroc = self.auroc('classwise')
		self.class_summary(auroc_item, class_auroc)

		auprc_item = QTreeWidgetItem(parent)
		auprc_item.setText(0, 'AUPRC')
		auprc_item.setText(1, str(self.auprc()))
		class_auprc = self.auprc('classwise')
		self.class_summary(auprc_item, class_auprc)

	def class_summary(self, parent, class_metric):
		"""
		Returns a summary of metrics with respect of each class.

		:param parent: the parent node in the tree view
		:type parent: QTreeWidgetItem
		:param class_metric: the classwise values of a particular metric
		:type class_metric: dict
		"""
		for c in class_metric:
			class_metric_item = QTreeWidgetItem(parent)
			class_metric_item.setText(0, str(c))
			class_metric_item.setToolTip(0, str(c))
			class_metric_item.setText(1, str(class_metric[c]))

	def plot_ROC(self, canvas):
		"""
		Plots ROC curves. This includes all classwise curves and a macro-averaged curve.

		Reference:
			http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

		:param canvas: the canvas where the plot is drawn
		:type canvas: FigureCanvas
		"""
		fpr = self.fpr_
		tpr = self.tpr_
		auc = self.class_auroc_
		classes = self.classes_
		num_classes = self.num_classes_

		rcParams.update({'font.size': 7})
		canvas.figure.clear()
		ax = canvas.figure.subplots()

		ax.plot(fpr['avg'], tpr['avg'], label='avg (area={0})'.format(self.avg_auroc_), \
			color = 'black', linewidth=2, linestyle='--')

		colors = cycle(['red', 'green', 'orange', 'blue', 'yellow', 'purple', 'cyan'])
		for i, color in zip(range(0, num_classes), colors):
			ax.plot(fpr[i], tpr[i], label='{0} (area={1})'.format(classes[i], auc[classes[i]]), \
				color=color, linewidth=1)

		ax.plot([0 ,1], [0, 1], color='lightgray', linewidth=1, linestyle='--')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.05])
		ax.set_xlabel('FPR')
		ax.set_ylabel('TPR')
		ax.legend(loc='lower right')

		canvas.figure.tight_layout()
		canvas.draw()

	def plot_precision_recall(self, canvas):
		"""
		Plots precision-recall curves. This includes all classwise curves and a macro-averaged curve.

		Reference:
			http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

		:param canvas: the canvas where the plot is drawn
		:type canvas: FigureCanva
		"""
		precision = self.precision_
		recall = self.recall_
		auc = self.class_auprc_
		classes = self.classes_
		num_classes = self.num_classes_

		rcParams.update({'font.size': 7})
		canvas.figure.clear()
		ax = canvas.figure.subplots()

		ax.plot(recall['avg'], precision['avg'], label='avg (area={0})'.format(self.avg_auprc_), \
			color = 'black', linewidth=2, linestyle='--')

		colors = cycle(['red', 'green', 'orange', 'blue', 'yellow', 'purple', 'cyan'])
		for i, color in zip(range(0, num_classes), colors):
			ax.plot(recall[i], precision[i], label='{0} (area={1})'.format(classes[i], auc[classes[i]]), \
				color=color, linewidth=1)

		# plot iso-f1 curves
		f1_scores = np.linspace(0.2, 0.8, num=4)
		x_top = np.array([0.1, 0.25, 0.45, 0.7])

		for i, f1_score in enumerate(f1_scores):
			x = np.linspace(0.01, 1)
			y = f1_score * x / (2 * x - f1_score)
			ax.plot(x[y >= 0], y[y >= 0], color='lightgray', alpha=0.2, linewidth=1)
			ax.annotate('f1={0:.1f}'.format(f1_score), xy=(x_top[i], 1)) 

		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.05])
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.legend(loc='lower right')

		canvas.figure.tight_layout()
		canvas.draw()

	def plot_confusion_matrix(self, canvas):
		"""
		Plots confusion matrix. The entries are normalized to lie between 0 and 1.

		Reference:
			http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

		:param canvas: the canvas where the plot is drawn
		:type canvas: FigureCanva
		"""
		classes = self.classes_
		num_classes = self.num_classes_
		cm = self.confusion_matrix_

		rcParams.update({'font.size': 7})
		canvas.figure.clear()
		ax = canvas.figure.subplots()
		tick_marks = np.arange(num_classes)

		ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
		ax.set_xticks(tick_marks)
		ax.set_yticks(tick_marks)
		ax.set_xticklabels(classes, rotation=45)
		ax.set_yticklabels(classes, rotation=45)

		cutoff = cm.max() / 2

		for i in range(0, cm.shape[0]):
			for j in range(0, cm.shape[1]):
				ax.text(j, i, cm[i, j], horizontalalignment='center', \
					color='white' if cm[i, j] > cutoff else 'black')

		ax.set_xlabel('Predicted class')
		ax.set_ylabel('True class')

		canvas.figure.tight_layout()
		canvas.draw()