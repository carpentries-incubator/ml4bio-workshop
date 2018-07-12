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
	def __init__(self, classifier, y, y_pred, y_prob, option):
		labels = classifier.classes_
		num_classes = len(labels)
		self.labels_ = labels
		self.num_classes_ = num_classes

		fpr = dict()
		tpr = dict()
		auroc = dict()

		precision = dict()
		recall = dict()
		auprc = dict()

		if option in ['holdout', 'loo']:
			self.accuracy_ = np.around(metrics.accuracy_score(y, y_pred), decimals=2)
			class_precision = metrics.precision_score(y, y_pred, average=None)
			class_recall = metrics.recall_score(y, y_pred, average=None)
			class_f1 = metrics.f1_score(y, y_pred, average=None)
			confusion_matrix = metrics.confusion_matrix(y, y_pred)

			for i in range(0, num_classes):
				fpr[i], tpr[i], _ = metrics.roc_curve(y, y_prob[:, i], pos_label=labels[i])
				auroc[labels[i]] = np.around(metrics.auc(fpr[i], tpr[i]), decimals=2)

				precision[i], recall[i], _ = metrics.precision_recall_curve(y, y_prob[:, i], pos_label=labels[i])
				precision[i] = np.flip(precision[i], 0)
				recall[i] = np.flip(recall[i], 0)
				auprc[labels[i]] = np.around(metrics.auc(recall[i], precision[i]), decimals=2)

		elif option == 'cv':
			k = len(y)
			accuracy = 0
			class_precision = np.zeros(num_classes)
			class_recall = np.zeros(num_classes)
			class_f1 = np.zeros(num_classes)
			confusion_matrix = np.zeros([num_classes, num_classes])

			for y_i, y_pred_i in zip(y, y_pred):
				accuracy += metrics.accuracy_score(y_i, y_pred_i)
				class_precision += metrics.precision_score(y_i, y_pred_i, average=None)
				class_recall += metrics.recall_score(y_i, y_pred_i, average=None)
				class_f1 += metrics.f1_score(y_i, y_pred_i, average=None)
				confusion_matrix += metrics.confusion_matrix(y_i, y_pred_i)

			self.accuracy_ = np.around(accuracy / k, decimals=2)
			class_precision = class_precision / k
			class_recall = class_recall / k
			class_f1 = class_f1 / k

			for i in range(0, num_classes):
				fpr_i = dict()
				tpr_i = dict()
				precision_i = dict()
				recall_i = dict()

				for j in range(0, k):
					fpr_i[j], tpr_i[j], _ = metrics.roc_curve(y[j], y_prob[j][:, i], pos_label=labels[i])
					precision_i[j], recall_i[j], _ = metrics.precision_recall_curve(y[j], y_prob[j][:, i], pos_label=labels[i])
				
				fpr[i] = np.unique(np.concatenate([fpr_i[j] for j in range(0, k)]))
				tpr[i] = np.zeros_like(fpr[i])
				recall[i] = np.unique(np.concatenate([recall_i[j] for j in range(0, k)]))
				precision[i] = np.zeros_like(recall[i])

				for j in range(0, k):
					tpr[i] += np.interp(fpr[i], fpr_i[j], tpr_i[j])
					precision[i] += np.interp(recall[i], np.flip(recall_i[j], 0), np.flip(precision_i[j], 0))
				
				tpr[i] /= k
				precision[i] /= k
				auroc[labels[i]] = np.around(metrics.auc(fpr[i], tpr[i]), decimals=2)
				auprc[labels[i]] = np.around(metrics.auc(recall[i], precision[i]), decimals=2)

		self.avg_precision_ = np.around(np.mean(class_precision), decimals=2)
		self.avg_recall_ = np.around(np.mean(class_recall), decimals=2)
		self.avg_f1_ = np.around(np.mean(class_f1), decimals=2)

		self.class_precision_ = dict(zip(labels, np.around(class_precision, decimals=2)))
		self.class_recall_ = dict(zip(labels, np.around(class_recall, decimals=2)))
		self.class_f1_ = dict(zip(labels, np.around(class_f1, decimals=2)))

		confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
		self.confusion_matrix_ = np.around(confusion_matrix, decimals=2)
		
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
		self.avg_auroc_ = np.around(metrics.auc(fpr['avg'], tpr['avg']), decimals=2)

		precision['avg'] = mean_precision
		recall['avg'] = all_recall
		self.avg_auprc_ = np.around(metrics.auc(recall['avg'], precision['avg']), decimals=2)

		self.fpr_ = fpr
		self.tpr_ = tpr
		self.class_auroc_ = auroc

		self.precision_ = precision
		self.recall_ = recall
		self.class_auprc_ = auprc

	# return accuracy of classifier
	def accuracy(self):
		return self.accuracy_

	# return precision of classifier
	def precision(self, option='average'):
		if option == 'classwise':	return self.class_precision_
		elif option == 'average':	return self.avg_precision_

	# return recall of classifier
	def recall(self, option='average'):
		if option == 'classwise':	return self.class_recall_
		elif option == 'average':	return self.avg_recall_

	# return f1-socre of classifier
	def f1(self, option='average'):
		if option == 'classwise':	return self.class_f1_
		elif option == 'average':	return self.avg_f1_

	# return AUROC of classifier
	def auroc(self, option='average'):
		if option == 'classwise':	return self.class_auroc_
		elif option == 'average':	return self.avg_auroc_

	# return AUPRC of classifier
	def auprc(self, option='average'):
		if option == 'classwise':	return self.class_auprc_
		elif option == 'average':	return self.avg_auprc_

	# return summary of metrics
	def summary(self, parent):
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

	# return summary of metrics w.r.t. individual classes
	def class_summary(self, parent, class_metric):
		for c in class_metric:
			class_metric_item = QTreeWidgetItem(parent)
			class_metric_item.setText(0, str(c))
			class_metric_item.setToolTip(0, str(c))
			class_metric_item.setText(1, str(class_metric[c]))

	# plot ROC curves
	#
	# reference:
	# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
	def plot_ROC(self, canvas):
		fpr = self.fpr_
		tpr = self.tpr_
		auc = self.class_auroc_
		labels = self.labels_
		num_classes = self.num_classes_

		rcParams.update({'font.size': 6})
		canvas.figure.clear()
		ax = canvas.figure.subplots()

		ax.plot(fpr['avg'], tpr['avg'], label='avg (area={0})'.format(self.avg_auroc_), \
			color = 'black', linewidth=2, linestyle='--')

		colors = cycle(['red', 'green', 'orange', 'blue', 'yellow', 'purple', 'cyan'])
		for i, color in zip(range(0, num_classes), colors):
			ax.plot(fpr[i], tpr[i], label='{0} (area={1})'.format(labels[i], auc[labels[i]]), \
				color=color, linewidth=1)

		ax.plot([0 ,1], [0, 1], color='lightgray', linewidth=1, linestyle='--')
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.05])
		ax.set_xlabel('FPR')
		ax.set_ylabel('TPR')
		ax.legend(loc='lower right')

		canvas.figure.tight_layout()
		canvas.draw()

	# plot precision-recall curves
	#
	# reference:
	# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
	def plot_precision_recall(self, canvas):
		precision = self.precision_
		recall = self.recall_
		auc = self.class_auprc_
		labels = self.labels_
		num_classes = self.num_classes_

		rcParams.update({'font.size': 6})
		canvas.figure.clear()
		ax = canvas.figure.subplots()

		ax.plot(recall['avg'], precision['avg'], label='avg (area={0})'.format(self.avg_auprc_), \
			color = 'black', linewidth=2, linestyle='--')

		colors = cycle(['red', 'green', 'orange', 'blue', 'yellow', 'purple', 'cyan'])
		for i, color in zip(range(0, num_classes), colors):
			ax.plot(recall[i], precision[i], label='{0} (area={1})'.format(labels[i], auc[labels[i]]), \
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

	# plot confusion matrix
	#
	# reference:
	# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	def plot_confusion_matrix(self, canvas):
		labels = self.labels_
		num_classes = self.num_classes_
		cm = self.confusion_matrix_

		rcParams.update({'font.size': 6})
		canvas.figure.clear()
		ax = canvas.figure.subplots()
		tick_marks = np.arange(num_classes)

		ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
		ax.set_xticks(tick_marks)
		ax.set_yticks(tick_marks)
		ax.set_xticklabels(labels, rotation=45)
		ax.set_yticklabels(labels, rotation=45)

		for i in range(0, cm.shape[0]):
			for j in range(0, cm.shape[1]):
				ax.text(j, i, cm[i, j], horizontalalignment='center', \
					color='white' if cm[i, j] > 0.5 else 'black')

		ax.set_xlabel('Predicted class')
		ax.set_ylabel('True class')

		canvas.figure.tight_layout()
		canvas.draw()