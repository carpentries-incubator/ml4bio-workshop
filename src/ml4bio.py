import os, sys, warnings
import pandas as pd
from pandas import errors
import numpy as np
from sklearn import tree, ensemble, neighbors, linear_model, neural_network, svm, naive_bayes, exceptions
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QPushButton, QRadioButton
from PyQt5.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QTextEdit, QLabel
from PyQt5.QtWidgets import QStackedWidget, QGroupBox, QFrame, QTableWidget, QTreeWidget, QTableWidgetItem, QTreeWidgetItem, QListView
from PyQt5.QtWidgets import QFormLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QFont, QIcon, QPixmap

from data import Data
from model import Model, DecisionTree, RandomForest, KNearestNeighbors, LogisticRegression, NeuralNetwork, SVM, NaiveBayes
from model_metrics import ModelMetrics

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.leftPanel = QStackedWidget(self)
        self.rightPanel = QGroupBox(self)
        self.initUI()

    # return a title
    def title(self, str, parent):
        label = QLabel(str, parent)
        font = QFont()
        font.setBold(True)
        label.setFont(font)
        return label

    # return a spin box
    def spin_box(self, val, min, max, stepsize, parent):
        box = QSpinBox(parent)
        box.setMinimum(min)
        box.setMaximum(max)
        box.setSingleStep(stepsize)
        box.setValue(val)
        return box

    # return a double spin box
    def double_spin_box(self, val, min, max, stepsize, prec, parent):
        box = QDoubleSpinBox(parent)
        box.setMinimum(min)
        box.setMaximum(max)
        box.setSingleStep(stepsize)
        box.setValue(val)
        box.setDecimals(prec)
        return box

    # load file
    def load(self, labeled):
        path = QFileDialog.getOpenFileName(self.dataPage)[0]
        if path != '':
            if labeled:
                try:
                    self.data = Data(path)
                # exception: wrong data format
                except:
                    self.error('format')
                    return
                # exception: too few samples
                if self.data.num_samples() < 20:
                    self.error('num_samples')
                    return

                self.labeledFileDisplay.setText(self.data.name('labeled'))
                self.unlabeledFilePushButton.setEnabled(True)
                self.splitFrame.setEnabled(True)
                self.validationFrame.setEnabled(True)
                self.dataNextPushButton.setEnabled(True)
            else:
                try:
                    self.data.add_unlabeled_data(path)
                # exception: wrong data format
                except errors.ParserError:
                    self.error('format')
                    return
                # exception: features do not match
                except:
                    self.error('features')
                    return

                self.unlabeledFileDisplay.setText(self.data.name('unlabeled'))
                self.predictionPushButton.setEnabled(True)

            self.data_summary()

    # show summary of data
    def data_summary(self):
        self.data.summary(self.data_summary_)

    # set decision tree hyperparameters
    def set_decision_tree(self):
        self.dtCriterionComboBox.setCurrentIndex(0)
        self.dtMaxDepthLineEdit.setText('None')
        self.dtMinSamplesSplitSpinBox.setValue(2)
        self.dtMinSamplesLeafSpinBox.setValue(1)
        self.dtClassWeightComboBox.setCurrentIndex(0)

    # set random forest hyperparameters
    def set_random_forest(self):
        self.rfCriterionComboBox.setCurrentIndex(0)
        self.rfNumEstimatorsSpinBox.setValue(10)
        self.rfMaxFeaturesComboBox.setCurrentIndex(0)
        self.rfMaxDepthLineEdit.setText('None')
        self.rfMinSamplesSplitSpinBox.setValue(2)
        self.rfMinSamplesLeafSpinBox.setValue(1)
        self.rfBootstrapCheckBox.setChecked(True)
        self.rfClassWeightComboBox.setCurrentIndex(0)

    # k-nearest neighbors hyperparameters
    def set_k_nearest_neighbors(self):
        self.knnNumNeighborsSpinBox.setValue(5)
        self.knnWeightsComboBox.setCurrentIndex(0)

        # discrete features: use hamming distance
        if self.data.feature_type() == 'discrete':
            self.knnMetricComboBox.setCurrentIndex(2)
            self.knnMetricListView.setRowHidden(0, True)
            self.knnMetricListView.setRowHidden(1, True)
        # continuous features: use euclidean or manhatten distance
        elif self.data.feature_type() == 'continuous':
            self.knnMetricListView.setRowHidden(2, True)
        # mixed features: DO NOT use KNN
        else:
            self.classTypeListView.setRowHidden(3, True)

    # set logistic regression hyperparameters
    def set_logistic_regression(self):
        self.lrRegularizationComboBox.setCurrentIndex(0)
        self.lrRglrStrengthLineEdit.setText('1.0')
        self.lrFitInterceptCheckBox.setChecked(True)
        self.lrInterceptScalingLineEdit.setText('1.0')
        self.lrSolverComboBox.setCurrentIndex(0)
        self.lrMultiClassComboBox.setCurrentIndex(0)
        self.lrClassWeightComboBox.setCurrentIndex(0)
        self.lrTolLineEdit.setText('1e-4')
        self.lrMaxIterLineEdit.setText('100')

    def update_logistic_regression(self):
        if self.lrSolverComboBox.currentIndex() in [1, 2, 4]:
            self.lrRegularizationListView.setRowHidden(1, True)
            self.lrRegularizationComboBox.setCurrentIndex(0)
        else:
            self.lrRegularizationListView.setRowHidden(1, False)
        if self.lrSolverComboBox.currentIndex() == 0:
            self.lrMultiClassListView.setRowHidden(1, True)
            self.lrMultiClassComboBox.setCurrentIndex(0)
        else:
            self.lrMultiClassListView.setRowHidden(1, False)

    # set neural network hyperparameters
    def set_neural_network(self):
        self.nnNumHiddenUnitsSpinBox.setValue(100)
        self.nnActivationComboBox.setCurrentIndex(0)
        self.nnSolverComboBox.setCurrentIndex(0)
        self.nnAlphaLineEdit.setText('1e-4')
        self.nnBatchSizeLineEdit.setText('20')
        self.nnLearningRateComboBox.setCurrentIndex(0)
        self.nnLearningRateInitLineEdit.setText('0.001')
        self.nnEarlyStoppingCheckBox.setChecked(False)
        self.nnTolLineEdit.setText('1e-4')
        self.nnMaxIterLineEdit.setText('200')

    def update_neural_network(self):
        if self.nnSolverComboBox.currentIndex() == 2:
            self.nnLearningRateComboBox.setEnabled(True)
        else:
            self.nnLearningRateComboBox.setDisabled(True)
        if self.nnSolverComboBox.currentIndex() == 1:
            self.nnLearningRateInitLineEdit.setDisabled(True)
            self.nnEarlyStoppingCheckBox.setDisabled(True)
        else:
            self.nnLearningRateInitLineEdit.setEnabled(True)
            self.nnEarlyStoppingCheckBox.setEnabled(True)

    # set svm hyperparameters
    def set_svm(self):
        self.svmPenaltyLineEdit.setText('1.0')
        self.svmKernelComboBox.setCurrentIndex(0)
        self.svmDegreeSpinBox.setValue(3)
        gamma = 1 / self.data.num_features('one-hot')
        self.svmGammaLineEdit.setText(str(round(gamma, 2)))
        self.svmCoefLineEdit.setText('0.0')
        self.svmClassWeightComboBox.setCurrentIndex(0)
        self.svmTolLineEdit.setText('1e-3')
        self.svmMaxIterLineEdit.setText('200')

    def update_svm(self):
        if self.svmKernelComboBox.currentIndex() == 2:
            self.svmDegreeSpinBox.setEnabled(True)
        else:
            self.svmDegreeSpinBox.setDisabled(True)
        if self.svmKernelComboBox.currentIndex() == 1:
            self.svmGammaLineEdit.setDisabled(True)
        else:
            self.svmGammaLineEdit.setEnabled(True)
        if self.svmKernelComboBox.currentIndex() in [0, 1]:
            self.svmCoefLineEdit.setDisabled(True)
        else:
            self.svmCoefLineEdit.setEnabled(True)

    # set naive bayes hyperparameters
    def set_naive_bayes(self):
        # discrete features: use multinomial NB
        if self.data.feature_type() == 'discrete':
            self.nbDistributionLabel.setText('multinomial')
        # continuous features: use gaussian NB
        elif self.data.feature_type() == 'continuous':
            self.nbDistributionLabel.setText('gaussian')
            self.nbAddSmoothDoubleSpinBox.setDisabled(True)
            self.nbFitPriorCheckBox.setDisabled(True)
        # mixed features: DO NOT use NB
        else:
            self.classTypeListView.setRowHidden(7, True)

        self.nbAddSmoothDoubleSpinBox.setValue(1.0)
        self.nbFitPriorCheckBox.setChecked(True)
        self.nbClassPriorLineEdit.setText('None')

    # reset hyperparameters of current classifier to default
    def reset(self):
        index = self.paramStack.currentIndex()
        if index == 1:      self.set_decision_tree()
        elif index == 2:    self.set_random_forest()
        elif index == 3:    self.set_k_nearest_neighbors()
        elif index == 4:    self.set_logistic_regression()
        elif index == 5:    self.set_neural_network()
        elif index == 6:    self.set_svm()
        elif index == 7:    self.set_naive_bayes()

        self.classNameLineEdit.clear()
        self.classCommentTextEdit.clear()

    # set up the system for training
    def set(self):
        # train/test split
        test_size = self.splitSpinBox.value() / 100
        stratify = self.splitCheckBox.isChecked()
        self.data.encode()
        self.data.split(test_size, stratify)

        # when data is 2-D and continuous, plot decision regions
        if self.data.feature_type() == 'continuous' \
            and self.data.num_features() == 2 and self.data.num_classes() == 2:
            self.dataPlotRadioButton.setEnabled(True)

        # validation method
        val = 0
        if self.holdoutRadioButton.isChecked(): 
            self.val_method = 'holdout'
            if self.holdoutSpinBox.value() == 0:
                val = self.warn('holdout')  # warning: no validation data
        elif self.cvRadioButton.isChecked():    
            self.val_method = 'cv'
            if self.data.num_samples() / self.cvSpinBox.value() < 10:
                val = self.warn('cv')       # warning: too few samples in each fold
        elif self.looRadioButton.isChecked():   
            self.val_method = 'loo'
            if self.data.num_samples() > 50:
                val = self.warn('loo')      # warning: too many samples for leave-one-out

        if val == QMessageBox.Close:
            return

        # classifier hyperparameters
        self.classTypeComboBox.setCurrentIndex(0)
        self.set_decision_tree()
        self.set_random_forest()
        self.set_k_nearest_neighbors()
        self.set_logistic_regression()
        self.set_neural_network()
        self.set_svm()
        self.set_naive_bayes()
        self.classNameLineEdit.clear()
        self.classCommentTextEdit.clear()

        # if no data is held out for validation, only show metrics on training data
        if self.holdoutRadioButton.isChecked() and self.holdoutSpinBox.value() == 0:
            self.performanceComboBox.setCurrentIndex(1)
            self.performanceListView.setRowHidden(0, True)

        self.leftPanel.setCurrentIndex(1)

    # back to default
    def clear(self, option):
        if option == 'all':
            self.clear('test')
            self.clear('train')
            self.data = None
            self.val_method = 'cv'
            self.tested = False
            self.unlabeledFilePushButton.setDisabled(True)
            self.labeledFileDisplay.setText('')
            self.unlabeledFileDisplay.setText('')
            self.data_summary_.clear()
            self.splitFrame.setDisabled(True)
            self.splitSpinBox.setValue(20)
            self.splitCheckBox.setChecked(True)
            self.validationFrame.setDisabled(True)
            self.holdoutSpinBox.setValue(20)
            self.holdoutSpinBox.setDisabled(True)
            self.cvRadioButton.setChecked(True)
            self.cvSpinBox.setValue(5)
            self.validationCheckBox.setChecked(True)
        if option == 'train':
            self.curr_model = None
            self.models.clear()
            self.models_table.setRowCount(0)
            self.model_summary_.clear()
            self.classNextPushButton.setDisabled(True)
            self.performanceListView.setRowHidden(0, False)
            self.performanceComboBox.setCurrentIndex(0)
            self.canvas.figure.clear()
            self.canvas.draw()
            self.dataPlotRadioButton.setDisabled(True)
            self.confusionMatrixRadioButton.setChecked(True)
            self.leftPanel.setCurrentIndex(0)
        elif option == 'test':
            self.selected_model = None
            self.bestPerformRadioButton.setChecked(True)
            self.metricComboBox.setCurrentIndex(0)
            self.leftPanel.setCurrentIndex(1)

    # train and return a decision tree classifier
    def decision_tree(self):
        data = self.data.train('integer')
        num_features = self.data.num_features('integer')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        max_depth = self.dtMaxDepthLineEdit.text()
        if max_depth == 'None':
            max_depth = None
        else:
            try:
                max_depth = int(max_depth)
            except:
                self.error('max_depth')
                return
            
            if max_depth <= 0 or float(self.dtMaxDepthLineEdit.text()) - max_depth != 0:
                self.error('max_depth')
                return

        class_weight = self.dtClassWeightComboBox.currentText()
        if class_weight == 'uniform':
            class_weight = None

        dt = tree.DecisionTreeClassifier(\
            criterion = self.dtCriterionComboBox.currentText(), \
            max_depth = max_depth, \
            min_samples_split = self.dtMinSamplesSplitSpinBox.value(), \
            min_samples_leaf = self.dtMinSamplesLeafSpinBox.value(), \
            class_weight = class_weight, \
            random_state = 0)

        return DecisionTree(dt, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())

    # train and return a random forest classifier
    def random_forest(self):
        data = self.data.train('integer')
        num_features = self.data.num_features('integer')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        max_depth = self.rfMaxDepthLineEdit.text()
        if max_depth == 'None':
            max_depth = None
        else:
            try:
                max_depth = int(max_depth)
            except:
                self.error('max_depth')
                return
            
            if max_depth <= 0 or float(self.rfMaxDepthLineEdit.text()) - max_depth != 0:
                self.error('max_depth')
                return

        max_features = self.rfMaxFeaturesComboBox.currentText()
        if max_features == 'all':
            max_features = None

        class_weight = self.rfClassWeightComboBox.currentText()
        if class_weight == 'uniform':
            class_weight = None

        rf = ensemble.RandomForestClassifier(\
            n_estimators = self.rfNumEstimatorsSpinBox.value(), \
            criterion = self.rfCriterionComboBox.currentText(), \
            max_depth = max_depth, \
            min_samples_split = self.rfMinSamplesSplitSpinBox.value(), \
            min_samples_leaf = self.rfMinSamplesLeafSpinBox.value(), \
            max_features = max_features, \
            bootstrap = self.rfBootstrapCheckBox.isChecked(), \
            class_weight = class_weight, \
            random_state = 0)
        
        return RandomForest(rf, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())

    # train and return a k-nearest neighbors classifier
    def k_nearest_neighbors(self):
        data = self.data.train('integer')
        num_features = self.data.num_features('integer')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        knn = neighbors.KNeighborsClassifier(\
            n_neighbors = self.knnNumNeighborsSpinBox.value(), \
            weights = self.knnWeightsComboBox.currentText(), \
            metric = self.knnMetricComboBox.currentText(), \
            algorithm = 'auto')
        
        return KNearestNeighbors(knn, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())

    # train and return a logistic regression classifier
    def logistic_regression(self):
        data = self.data.train('one-hot')
        num_features = self.data.num_features('one-hot')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        try:
            penalty = float(self.lrRglrStrengthLineEdit.text())
        except:
            self.error('penalty')
            return
        try:
            intercept_scaling = float(self.lrInterceptScalingLineEdit.text())
        except:
            self.error('intercept_scaling')
            return
        try:
            tol = float(self.lrTolLineEdit.text())
        except:
            self.error('tol')
            return
        try:
            max_iter = int(self.lrMaxIterLineEdit.text())
        except:
            self.error('max_iter')
            return

        if penalty <= 0:
            self.error('penalty')
            return
        if tol <= 0:
            self.error('tol')
            return
        if max_iter <= 0 or float(self.lrMaxIterLineEdit.text()) - max_iter != 0:
            self.error('max_iter')
            return

        class_weight = self.lrClassWeightComboBox.currentText()
        if class_weight == 'uniform':
            class_weight = None

        lr = linear_model.LogisticRegression(\
            penalty = self.lrRegularizationComboBox.currentText(), \
            tol = tol, \
            C = penalty, \
            fit_intercept = self.lrFitInterceptCheckBox.isChecked(), \
            intercept_scaling = intercept_scaling, \
            class_weight = class_weight, \
            solver = 'saga', \
            max_iter = max_iter, \
            multi_class = self.lrMultiClassComboBox.currentText(), \
            random_state = 0)
        
        try:
            model =  LogisticRegression(lr, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())
        except exceptions.ConvergenceWarning:
            self.info('converge')
            return

        return model

    # train and return a neural network classifier
    def neural_network(self):
        data = self.data.train('one-hot')
        num_features = self.data.num_features('one-hot')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        try:
            penalty = float(self.nnAlphaLineEdit.text())
        except:
            self.error('penalty')
            return
        try:
            batch_size = int(self.nnBatchSizeLineEdit.text())
        except:
            self.error('batch_size')
            return
        try:
            learning_rate_init = float(self.nnLearningRateInitLineEdit.text())
        except:
            self.error('learning_rate_init')
            return
        try:
            tol = float(self.nnTolLineEdit.text())
        except:
            self.error('tol')
            return
        try:
            max_iter = int(self.nnMaxIterLineEdit.text())
        except:
            self.error('max_iter')
            return

        if penalty <= 0:
            self.error('penalty')
            return
        if batch_size <= 0 or float(self.nnBatchSizeLineEdit.text()) - batch_size != 0:
            self.error('batch_size')
            return
        if learning_rate_init <= 0:
            self.error('learning_rate_init')
            return
        if tol <= 0:
            self.error('tol')
            return
        if max_iter <= 0 or float(self.nnMaxIterLineEdit.text()) - max_iter != 0:
            self.error('max_iter')
            return

        nn = neural_network.MLPClassifier(\
            hidden_layer_sizes = self.nnNumHiddenUnitsSpinBox.value(), \
            activation = self.nnActivationComboBox.currentText(), \
            solver = self.nnSolverComboBox.currentText(), \
            alpha = penalty, \
            batch_size = batch_size, \
            learning_rate = self.nnLearningRateComboBox.currentText(), \
            learning_rate_init = learning_rate_init, \
            max_iter = max_iter, \
            tol = tol, \
            early_stopping = self.nnEarlyStoppingCheckBox.isChecked(), \
            random_state = 0)

        try:
            model = NeuralNetwork(nn, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())
        except exceptions.ConvergenceWarning:
            self.info('converge')
            return

        return model

    # train and return a svm classifier
    def svm(self):
        data = self.data.train('one-hot')
        num_features = self.data.num_features('one-hot')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        try:
            penalty = float(self.svmPenaltyLineEdit.text())
        except:
            self.error('penalty')
            return
        try:
            gamma = float(self.svmGammaLineEdit.text())
        except:
            self.error('kernel_coef')
            return
        try:
            coef0 = float(self.svmCoefLineEdit.text())
        except:
            self.error('indenpendent_term')
            return
        try:
            tol = float(self.svmTolLineEdit.text())
        except:
            self.error('tol')
            return
        try:
            max_iter = int(self.svmMaxIterLineEdit.text())
        except:
            self.error('max_iter')
            return

        if penalty <= 0:
            self.error('penalty')
            return
        if gamma <= 0:
            self.error('kernel_coef')
            return
        if tol <= 0:
            self.error('tol')
            return
        if max_iter <= 0 or float(self.svmMaxIterLineEdit.text()) - max_iter != 0:
            self.error('max_iter')
            return

        class_weight = self.svmClassWeightComboBox.currentText()
        if class_weight == 'uniform':
            class_weight = None

        svc = svm.SVC(\
            C = penalty, \
            kernel = self.svmKernelComboBox.currentText(), \
            degree = self.svmDegreeSpinBox.value(), \
            gamma = gamma, \
            coef0 = coef0, \
            probability = True, \
            tol = tol, \
            class_weight = class_weight, \
            max_iter = max_iter, \
            random_state = 0)
           
        try: 
            model = SVM(svc, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked())
        except exceptions.ConvergenceWarning:
            self.info('converge')
            return

        return model

    # train and return a naive bayes classifier
    def naive_bayes(self):
        data = self.data.train('integer')
        num_features = self.data.num_features('integer')
        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]

        class_prior = self.nbClassPriorLineEdit.text()
        if class_prior == 'None':
            class_prior = None
        else:
            try:
                class_prior = [float(i.strip()) for i in class_prior.split(',')]
            except:
                self.error('class_prior')
                return

            if len(class_prior) != self.data.num_classes() or sum(class_prior) != 1:
                self.error('class_prior')
                return

        if self.nbDistributionLabel.text() == 'multinomial':
            nb = naive_bayes.MultinomialNB(\
                alpha = self.nbAddSmoothDoubleSpinBox.value(), \
                fit_prior = self.nbFitPriorCheckBox.isChecked(), \
                class_prior = class_prior)
            mode = 'multinomial'
        elif self.nbDistributionLabel.text() == 'gaussian':
            nb = naive_bayes.GaussianNB(priors = class_prior)
            mode = 'gaussian'
        
        return NaiveBayes(nb, X, y, val_method=self.val_method, \
                val_size=self.holdoutSpinBox.value() / 100, \
                k = self.cvSpinBox.value(), \
                stratify=self.validationCheckBox.isChecked(), \
                mode=mode)

    # train a classifier and push it into the model table
    def train(self):
        index = self.paramStack.currentIndex()
        if index == 1:      model = self.decision_tree()
        elif index == 2:    model = self.random_forest()
        elif index == 3:    model = self.k_nearest_neighbors()
        elif index == 4:    model = self.logistic_regression()
        elif index == 5:    model = self.neural_network()
        elif index == 6:    model = self.svm()
        elif index == 7:    model = self.naive_bayes()

        # training failed (e.g. invalid hyperparameters or model did not converge)
        if model is None:
            return

        name = self.classNameLineEdit.text().strip()
        if name != '':
            if name in self.models:
                self.error('name')
                return
            model.set_name(name)

        model.set_comment(self.classCommentTextEdit.toPlainText().strip())
        self.classNameLineEdit.clear()
        self.classCommentTextEdit.clear()

        self.models[model.name()] = model   # add to the collection of classifiers
        if self.performanceComboBox.currentIndex() == 0:
            self.push(model, 'val')
        else:
            self.push(model, 'train')

        self.classNextPushButton.setEnabled(True)

    # enter a classifier into the table
    def push(self, model, option):
        if option == 'train':
            metrics = model.metrics('train')
        else:
            metrics = model.metrics('val')
            if metrics is None:
                return

        row = self.models_table.rowCount()  # enter as the last row
        self.models_table.insertRow(row)

        name_item = QTableWidgetItem(model.name())
        type_item = QTableWidgetItem(model.type())
        accuracy_item = QTableWidgetItem()
        precision_item = QTableWidgetItem()
        recall_item = QTableWidgetItem()
        f1_item = QTableWidgetItem()
        auroc_item = QTableWidgetItem()
        auprc_item = QTableWidgetItem()

        name_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        accuracy_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        precision_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        recall_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        f1_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        auroc_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        auprc_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        self.models_table.setItem(row, 0, name_item)
        self.models_table.setItem(row, 1, type_item)
        self.models_table.setItem(row, 2, accuracy_item)
        self.models_table.setItem(row, 3, precision_item)
        self.models_table.setItem(row, 4, recall_item)
        self.models_table.setItem(row, 5, f1_item)
        self.models_table.setItem(row, 6, auroc_item)
        self.models_table.setItem(row, 7, auprc_item)

        self.set_metrics(row, metrics)
        self.switch_model(row, 0)    # mark the new classifier as selected

    # set metrics of a classifier in the table
    def set_metrics(self, row, metrics):
        self.models_table.item(row, 2).setText(str(metrics.accuracy()))
        self.models_table.item(row, 3).setText(str(metrics.precision()))
        self.models_table.item(row, 4).setText(str(metrics.recall()))
        self.models_table.item(row, 5).setText(str(metrics.f1()))
        self.models_table.item(row, 6).setText(str(metrics.auroc()))
        self.models_table.item(row, 7).setText(str(metrics.auprc()))

    # switch between metrics on training and validation data
    def switch_metrics(self, index):
        if index == 0:      option = 'val'
        elif index == 1:    option = 'train'
        else:               option = 'test'

        for i in range(0, self.models_table.rowCount()):
            model = self.models[self.models_table.item(i, 0).text()]
            metrics = model.metrics(option)
            if metrics is not None:
                self.set_metrics(i, metrics)
                self.models_table.showRow(i)
            else:
                self.models_table.hideRow(i)

        if self.curr_model is not None:
            row = 0
            for i in range(0, self.models_table.rowCount()):
                if self.models_table.item(i, 0).text() == self.curr_model.name():
                    row = i

            # if classifier has not been tested, clear model summary and plots
            # (useful when user switches from train/validation metrics to test metrics)
            if self.models_table.isRowHidden(row):
                self.model_summary_.clear()
                self.canvas.figure.clear()
                self.canvas.draw()
            else:
                self.plot()

        if self.leftPanel.currentIndex() == 2 and self.performanceComboBox.currentIndex() != 2:
            self.select('best')
            self.select('user')

    # switch classifier, show summary and plots
    def switch_model(self, row, col):
        self.models_table.setCurrentCell(row, 0)
        self.curr_model = self.models[self.models_table.item(row, 0).text()]
        self.model_summary()
        self.plot()

        # respond to mouse click when the classifier to be tested is user-picked
        if self.leftPanel.currentIndex() == 2 and self.userPickRadioButton.isChecked():
            self.selected_model = self.curr_model
            self.userPickLabel.setText(self.selected_model.name())
            self.testPushButton.setEnabled(True)

    # show summary of classifier
    def model_summary(self):
            self.curr_model.summary(self.model_summary_)

    # show plots
    def plot(self):
        if self.curr_model is not None:
            index = self.performanceComboBox.currentIndex()
            if  index == 0:
                metrics = self.curr_model.metrics('val')
                option = 'train'
            elif index == 1:
                metrics = self.curr_model.metrics('train')
                option = 'train'
            else:
                metrics = self.curr_model.metrics('test')
                option = 'test'

            if self.confusionMatrixRadioButton.isChecked():
                metrics.plot_confusion_matrix(self.canvas)
            elif self.rocRadioButton.isChecked():
                metrics.plot_ROC(self.canvas)
            elif self.prRadioButton.isChecked():
                metrics.plot_precision_recall(self.canvas)
            elif self.dataPlotRadioButton.isChecked():
                self.curr_model.plot_decision_regions(option, self.canvas)

    # sort classifiers in descending order w.r.t. the selected metric
    def sort(self, col):
        self.models_table.sortItems(col, Qt.DescendingOrder)

    # select a model for testing
    def select(self, option):
        if option == 'best' and self.bestPerformRadioButton.isChecked():
            index = self.metricComboBox.currentIndex()
            if index == 0:
                self.selected_model = None
                self.userPickLabel.setText('None')
                self.testPushButton.setDisabled(True)
                return
            else:
                col = index + 1
                self.sort(col)
                self.switch_model(0, 0)
                self.selected_model = self.curr_model        
                self.userPickLabel.setText(self.selected_model.name())
                self.testPushButton.setEnabled(True)
        elif option == 'user' and self.userPickRadioButton.isChecked():
            self.selected_model = self.curr_model        
            self.userPickLabel.setText(self.selected_model.name())
            self.testPushButton.setEnabled(True)

    # test the selected classifier
    def test(self):
        # when at least one classifier has been tested
        if self.tested:
            val = self.warn('test')
            if val == QMessageBox.Cancel:
                return
        # when no classifier has been tested
        else:
            msg = 'You cannot train more classifiers once you begin testing. '
            msg += 'Do you want to proceed?'
            self.test_box.setText(msg)
            val = self.test_box.exec_()
            if val == QMessageBox.Cancel:
                return

        if self.selected_model.type() in ['decision tree', 'random forest', \
            'k-nearest neighbors', 'naive bayes']:
            data = self.data.test('integer')
            num_features = self.data.num_features('integer')
        else:
            data = self.data.test('one-hot')
            num_features = self.data.num_features('one-hot')

        X = data.iloc[:, 0: num_features]
        y = data.iloc[:, num_features]
        self.selected_model.test(X, y)
        self.tested = True

        row = 0
        for i in range(0, self.models_table.rowCount()):
            if self.models_table.item(i, 0).text() == self.selected_model.name():
                row = i

        self.switch_model(row, 0)       # swtich to the selected model
        self.performanceListView.setRowHidden(2, False)
        self.bestPerformRadioButton.setChecked(True)
        self.metricComboBox.setCurrentIndex(0)          # DO NOT switch order
        self.performanceComboBox.setCurrentIndex(2)     # (bug in GUI library) 

        self.testBackPushButton.setDisabled(True)
        self.testFinishPushButton.setEnabled(True)

    # use the selected classifier to make prediction
    def predict(self):
        if self.curr_model.type() in ['decision tree', 'random forest', \
            'k-nearest neighbors', 'naive bayes']:
            data = self.data.prediction('integer')
        else:
            data = self.data.prediction('one-hot')

        path = QFileDialog.getSaveFileName(self.testPage)[0]
        self.curr_model.predict(data, path)

    # finish analyzing the dataset
    def finish(self):
        msg = 'Do you want to analyze more data?'
        self.finish_box.setText(msg)
        val = self.finish_box.exec_()
        if val == QMessageBox.Yes:
            self.clear('all')
        elif val == QMessageBox.No:
            QApplication.instance().quit()

    # show reminders
    def info(self, flag):
        if flag == 'converge':
            msg = 'Maximum iterations reached and the classifier has not converged yet. '
            msg += 'Consider increasing max_iter.'
        self.info_box.setText(msg)
        self.info_box.exec_()

    # show warning messages
    def warn(self, flag):
        self.warn_message(flag)
        return self.warn_box.exec_()

    # show error messages
    def error(self, flag):
        self.error_message(flag)
        return self.err_box.exec_()

    # construct warning messages
    def warn_message(self, flag):
        if flag == 'holdout':
            msg = 'No validation data is allocated. Classifiers may overfit.'
        elif flag == 'cv':
            msg = 'Too few samples in each fold. '
            msg += 'Use fewer folds or consider leave-one-out validation.'
        elif flag == 'loo':
            msg = 'Too many samples for leave-one-out validation. '
            msg += 'Consider hold-out validation or k-fold cross-validation.'
        elif flag == 'test':
            msg = 'Model selection on test data may lead to an overfit model.'
        self.warn_box.setText(msg)

    # construct error messages
    def error_message(self, flag):
        if flag == 'format':
            msg = 'Wrong data format. Only .csv is accepted.'
        elif flag == 'num_samples':
            msg = 'Too few samples. At least 20 samples are required.'
        elif flag == 'features':
            msg = 'Feature names do not match.'
        elif flag == 'max_depth':
            msg = 'max_depth: must be a positive integer (e.g. 3) or None'
        elif flag == 'penalty':
            msg = 'penalty: must be a positive number (e.g. 0.5)'
        elif flag == 'intercept_scaling':
            msg = 'intercept_scaling: must be a number (e.g. 1.0)'
        elif flag == 'batch_size':
            msg = 'batch_size: must be a positive integer (e.g. 200)'
        elif flag == 'learning_rate_init':
            msg = 'learning_rate_init: must be a positive number (e.g. 0.001)'
        elif flag == 'kernel_coef':
            msg = 'kernel_coef: must be a positive number (e.g. 0.5)'
        elif flag == 'indenpendent_term':
            msg = 'indenpendent_term: must be a number (e.g. 0)'
        elif flag == 'class_prior':
            msg = 'class_prior: must be None or follow the format <value>,...,<value> '
            msg += 'where the number of values equals the number of classes '
            msg += 'and the values sum to 1 (e.g. 0.2,0.6,0.2 for three classes)'
        elif flag == 'tol':
            msg = 'tol: must be a positive number (e.g. 1e-4)'
        elif flag == 'max_iter':
            msg = 'max_iter: must be a positive integer (e.g. 200)'
        elif flag == 'name':
            msg = 'Classifier name already exists.'
        self.err_box.setText(msg)

    def initUI(self):
        self.data = None
        self.val_method = 'cv'
        self.models = dict()
        self.curr_model = None      # classifier highlighted in the table
        self.selected_model = None  # classifier selected for testing
        self.tested = False

        self.info_box = QMessageBox()
        self.info_box.setIcon(QMessageBox.Information)
        self.info_box.setStandardButtons(QMessageBox.Ok)

        self.warn_box = QMessageBox()
        self.warn_box.setIcon(QMessageBox.Warning)
        self.warn_box.setStandardButtons(QMessageBox.Close | QMessageBox.Ignore)
        
        self.err_box = QMessageBox()
        self.err_box.setIcon(QMessageBox.Critical)
        self.err_box.setStandardButtons(QMessageBox.Ok)

        self.test_box = QMessageBox()
        self.test_box.setIcon(QMessageBox.Question)
        self.test_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)

        self.finish_box = QMessageBox()
        self.finish_box.setIcon(QMessageBox.Question)
        self.finish_box.setStandardButtons(QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes)

        self.font = QFont()         # font for table and summary
        self.font.setPointSize(10)

        ########## 1st page: data loading and splitting ##########
        self.dataPage = QWidget()
        self.openIcon = QIcon('icons/open.png')
        self.dataPageLayout = QVBoxLayout(self.dataPage)

        ### load labeled data
        self.labeledDataLabel = self.title('Labeled Data (.csv):', self.dataPage)
        self.labeledFilePushButton = QPushButton('Select File...', self.dataPage)
        self.labeledFilePushButton.setIcon(self.openIcon)
        self.labeledFilePushButton.setMaximumWidth(140)
        self.labeledFileSpacer = QSpacerItem(40, 20)
        self.labeledFileDisplay = QLabel('', self.dataPage)

        self.dataPageLayout.addWidget(self.labeledDataLabel)
        self.labeledFileLayout = QHBoxLayout()
        self.labeledFileLayout.addWidget(self.labeledFilePushButton)
        self.labeledFileLayout.addItem(self.labeledFileSpacer)
        self.labeledFileLayout.addWidget(self.labeledFileDisplay)
        self.dataPageLayout.addLayout(self.labeledFileLayout)

        ### load unlabeled data
        self.unlabeledDataLabel = self.title('Unlabeled Data (.csv):', self.dataPage)
        self.unlabeledFilePushButton = QPushButton('Select File...', self.dataPage)
        self.unlabeledFilePushButton.setIcon(self.openIcon)
        self.unlabeledFilePushButton.setMaximumWidth(140)
        self.unlabeledFilePushButton.setDisabled(True)
        self.unlabeledFileSpacer = QSpacerItem(40, 20)
        self.unlabeledFileDisplay = QLabel('', self.dataPage)

        self.dataPageLayout.addWidget(self.unlabeledDataLabel)
        self.unlabeledFileLayout = QHBoxLayout()
        self.unlabeledFileLayout.addWidget(self.unlabeledFilePushButton)
        self.unlabeledFileLayout.addItem(self.unlabeledFileSpacer)
        self.unlabeledFileLayout.addWidget(self.unlabeledFileDisplay)
        self.dataPageLayout.addLayout(self.unlabeledFileLayout)

        ### data summary
        self.dataSummaryLabel = self.title('Data Summary:', self.dataPage)
        self.data_summary_ = QTreeWidget(self.dataPage)
        self.data_summary_.setColumnCount(2)
        self.data_summary_.setHeaderHidden(True)
        self.data_summary_.setColumnWidth(0, 200)
        self.dataPageLayout.addWidget(self.dataSummaryLabel)
        self.dataPageLayout.addWidget(self.data_summary_)

        ### train/test split
        self.trainTestLabel = self.title('Train/Test Split:', self.dataPage)
        self.splitFrame = QFrame(self.dataPage)
        self.splitFrame.setAutoFillBackground(True)
        self.splitFrame.setDisabled(True)
        self.splitSpinBox = self.spin_box(20, 10, 50, 10, self.splitFrame)
        self.splitSpinBox.setMaximumWidth(60)
        self.splitLabel = QLabel('% for test', self.splitFrame)
        self.splitCheckBox = QCheckBox('Stratified Sampling', self.splitFrame)
        self.splitCheckBox.setChecked(True)

        self.dataPageLayout.addWidget(self.trainTestLabel)
        self.trainTestLayout = QHBoxLayout()
        self.trainTestLayout.addWidget(self.splitSpinBox)
        self.trainTestLayout.addWidget(self.splitLabel)
        self.trainTestLayout.addWidget(self.splitCheckBox)
        self.splitLayout = QVBoxLayout(self.splitFrame)
        self.splitLayout.addLayout(self.trainTestLayout)
        self.dataPageLayout.addWidget(self.splitFrame)

        ### validation methodology
        self.validationLabel = self.title('Validation Methodology:', self.dataPage)
        self.validationFrame = QFrame(self.dataPage)
        self.validationFrame.setAutoFillBackground(True)
        self.validationFrame.setDisabled(True)
        self.holdoutRadioButton = QRadioButton('Holdout Validation', self.validationFrame)
        self.cvRadioButton = QRadioButton('K-Fold Cross-Validation', self.validationFrame)
        self.cvRadioButton.setChecked(True)
        self.looRadioButton = QRadioButton('Leave-One-Out Validation', self.validationFrame)
        self.holdoutSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.cvSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.looSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.validationSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.holdoutSpinBox = self.spin_box(20, 0, 50, 10, self.validationFrame)
        self.holdoutSpinBox.setMaximumWidth(50)
        self.holdoutSpinBox.setDisabled(True)
        self.cvSpinBox = self.spin_box(5, 2, 10, 1, self.validationFrame)
        self.validationCheckBox = QCheckBox('Stratified Sampling', self.validationFrame)
        self.validationCheckBox.setChecked(True)
        self.holdoutLabel = QLabel('% for validation', self.validationFrame)
        self.cvLabel = QLabel('folds', self.validationFrame)

        self.dataPageLayout.addWidget(self.validationLabel)
        self.holdoutLayout = QHBoxLayout()
        self.holdoutLayout.addWidget(self.holdoutRadioButton)
        self.holdoutLayout.addItem(self.holdoutSpacer)
        self.holdoutLayout.addWidget(self.holdoutSpinBox)
        self.holdoutLayout.addWidget(self.holdoutLabel)
        self.cvLayout = QHBoxLayout()
        self.cvLayout.addWidget(self.cvRadioButton)
        self.cvLayout.addItem(self.cvSpacer)
        self.cvLayout.addWidget(self.cvSpinBox)
        self.cvLayout.addWidget(self.cvLabel)
        self.looLayout = QHBoxLayout()
        self.looLayout.addWidget(self.looRadioButton)
        self.looLayout.addItem(self.looSpacer)
        self.samplingLayout = QHBoxLayout()
        self.samplingLayout.addItem(self.validationSpacer)
        self.samplingLayout.addWidget(self.validationCheckBox)
        self.validationLayout = QVBoxLayout(self.validationFrame)
        self.validationLayout.addLayout(self.holdoutLayout)
        self.validationLayout.addLayout(self.cvLayout)
        self.validationLayout.addLayout(self.looLayout)
        self.validationLayout.addLayout(self.samplingLayout)
        self.dataPageLayout.addWidget(self.validationFrame)

        ### data spacer and line
        self.dataSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.dataPageLayout.addItem(self.dataSpacer)
        self.dataLine = QFrame(self.dataPage)
        self.dataLine.setFrameShape(QFrame.HLine)
        self.dataLine.setFrameShadow(QFrame.Sunken)
        self.dataPageLayout.addWidget(self.dataLine)

        ### next button
        self.dataNextSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.dataNextPushButton = QPushButton('Next', self.dataPage)
        self.dataNextPushButton.setDefault(True)
        self.dataNextPushButton.setMinimumWidth(90)
        self.dataNextPushButton.setDisabled(True)
        self.dataNextLayout = QHBoxLayout()
        self.dataNextLayout.addItem(self.dataNextSpacer)
        self.dataNextLayout.addWidget(self.dataNextPushButton)
        self.dataPageLayout.addLayout(self.dataNextLayout)

        self.leftPanel.addWidget(self.dataPage)

        # signal handling
        self.labeledFilePushButton.clicked.connect(lambda: self.load(labeled=True))
        self.unlabeledFilePushButton.clicked.connect(lambda: self.load(labeled=False))

        self.holdoutRadioButton.toggled.connect(self.holdoutSpinBox.setEnabled)
        self.holdoutRadioButton.toggled.connect(self.cvSpinBox.setDisabled)
        self.holdoutRadioButton.toggled.connect(self.validationCheckBox.setEnabled)
        self.cvRadioButton.toggled.connect(self.holdoutSpinBox.setDisabled)
        self.cvRadioButton.toggled.connect(self.cvSpinBox.setEnabled)
        self.cvRadioButton.toggled.connect(self.validationCheckBox.setEnabled)
        self.looRadioButton.toggled.connect(self.holdoutSpinBox.setDisabled)
        self.looRadioButton.toggled.connect(self.cvSpinBox.setDisabled)
        self.looRadioButton.toggled.connect(self.validationCheckBox.setDisabled)

        self.dataNextPushButton.clicked.connect(self.set)
        
        ########## 2nd page: training ##########

        self.modelPage = QWidget()
        self.modelPageLayout = QVBoxLayout(self.modelPage)

        ### classifier type
        self.classTypeLabel = self.title('Classifier Type:', self.modelPage)
        self.classTypeComboBox = QComboBox(self.modelPage)
        self.classTypeListView = self.classTypeComboBox.view()
        self.classTypeLayout = QHBoxLayout()
        self.classTypeLayout.addWidget(self.classTypeLabel)
        self.classTypeLayout.addWidget(self.classTypeComboBox)

        ### classifier parameter stack
        self.paramStack = QStackedWidget(self.modelPage)
        self.paramStack.setMinimumHeight(320)
        self.modelPageLayout.addLayout(self.classTypeLayout)
        self.modelPageLayout.addWidget(self.paramStack)

        ## initial empty page
        self.noneIcon = QIcon('icons/none.png')
        self.classTypeComboBox.addItem(self.noneIcon, '-------- select --------')
        self.initPage = QWidget()
        self.paramStack.addWidget(self.initPage)

        ## decision tree
        self.dtIcon = QIcon('icons/dt.png')
        self.classTypeComboBox.addItem(self.dtIcon, 'Decision Tree')
        self.dtPage = QWidget()
        self.paramStack.addWidget(self.dtPage)

        # fields
        self.dtCriterionComboBox = QComboBox(self.dtPage)
        self.dtCriterionComboBox.addItem('gini')
        self.dtCriterionComboBox.addItem('entropy')
        self.dtMaxDepthLineEdit = QLineEdit('None', self.dtPage)
        self.dtMinSamplesSplitSpinBox = self.spin_box(2, 2, 20, 1, self.dtPage)
        self.dtMinSamplesLeafSpinBox = self.spin_box(1, 1, 20, 1, self.dtPage)
        self.dtClassWeightComboBox = QComboBox(self.dtPage)
        self.dtClassWeightComboBox.addItem('uniform')
        self.dtClassWeightComboBox.addItem('balanced')

        # layout
        self.dtLayout = QFormLayout()
        self.dtLayout.addRow('criterion:', self.dtCriterionComboBox)
        self.dtLayout.addRow('max_depth:', self.dtMaxDepthLineEdit)
        self.dtLayout.addRow('min_samples_split:', self.dtMinSamplesSplitSpinBox)
        self.dtLayout.addRow('min_samples_leaf:', self.dtMinSamplesLeafSpinBox)
        self.dtLayout.addRow('class_weight:', self.dtClassWeightComboBox)

        self.dtPageLayout = QVBoxLayout(self.dtPage)
        self.dtPageLayout.addLayout(self.dtLayout)

        ## Random forest
        self.rfIcon = QIcon('icons/rf.png')
        self.classTypeComboBox.addItem(self.rfIcon, 'Random Forest')
        self.rfPage = QWidget()
        self.paramStack.addWidget(self.rfPage)

        # fields
        self.rfCriterionComboBox = QComboBox(self.rfPage)
        self.rfCriterionComboBox.addItem('gini')
        self.rfCriterionComboBox.addItem('entropy')
        self.rfNumEstimatorsSpinBox = self.spin_box(10, 2, 20, 1, self.rfPage)
        self.rfMaxFeaturesComboBox = QComboBox(self.rfPage)
        self.rfMaxFeaturesComboBox.addItem('sqrt')
        self.rfMaxFeaturesComboBox.addItem('log2')
        self.rfMaxFeaturesComboBox.addItem('all')
        self.rfMaxDepthLineEdit = QLineEdit('None', self.rfPage)
        self.rfMinSamplesSplitSpinBox = self.spin_box(2, 2, 20, 1, self.rfPage)
        self.rfMinSamplesLeafSpinBox = self.spin_box(1, 1, 20, 1, self.rfPage)
        self.rfBootstrapCheckBox = QCheckBox('', self.rfPage)
        self.rfBootstrapCheckBox.setChecked(True)
        self.rfClassWeightComboBox = QComboBox(self.rfPage)
        self.rfClassWeightComboBox.addItem('uniform')
        self.rfClassWeightComboBox.addItem('balanced')

        # layout
        self.rfLayout = QFormLayout()
        self.rfLayout.addRow('criterion:', self.rfCriterionComboBox)
        self.rfLayout.addRow('n_estimators:', self.rfNumEstimatorsSpinBox)
        self.rfLayout.addRow('max_features:', self.rfMaxFeaturesComboBox)
        self.rfLayout.addRow('max_depth:', self.rfMaxDepthLineEdit)
        self.rfLayout.addRow('min_samples_split:', self.rfMinSamplesSplitSpinBox)
        self.rfLayout.addRow('min_samples_leaf:', self.rfMinSamplesLeafSpinBox)
        self.rfLayout.addRow('bootstrap:', self.rfBootstrapCheckBox)
        self.rfLayout.addRow('class_weight:', self.rfClassWeightComboBox)

        self.rfPageLayout = QVBoxLayout(self.rfPage)
        self.rfPageLayout.addLayout(self.rfLayout)

        ## K-nearest neighbors
        self.knnIcon = QIcon('icons/knn.png')
        self.classTypeComboBox.addItem(self.knnIcon, 'K-Nearest Neighbors')
        self.knnPage = QWidget()
        self.paramStack.addWidget(self.knnPage)

        # fields
        self.knnNumNeighborsSpinBox = self.spin_box(5, 1, 20, 1, self.knnPage)
        self.knnWeightsComboBox = QComboBox(self.knnPage)
        self.knnWeightsComboBox.addItem('uniform')
        self.knnWeightsComboBox.addItem('distance')
        self.knnMetricComboBox = QComboBox(self.knnPage)
        self.knnMetricListView =self.knnMetricComboBox.view()
        self.knnMetricComboBox.addItem('euclidean') #
        self.knnMetricComboBox.addItem('manhattan') # DO NOT change order
        self.knnMetricComboBox.addItem('hamming')   #

        # layout
        self.knnLayout = QFormLayout()
        self.knnLayout.addRow('n_neighbors:', self.knnNumNeighborsSpinBox)
        self.knnLayout.addRow('weights:', self.knnWeightsComboBox)
        self.knnLayout.addRow('metric:', self.knnMetricComboBox)

        self.knnPageLayout = QVBoxLayout(self.knnPage)
        self.knnPageLayout.addLayout(self.knnLayout)

        ## Logistic regression
        self.lrIcon = QIcon('icons/lr.png')
        self.classTypeComboBox.addItem(self.lrIcon, 'Logistic Regression')
        self.lrPage = QWidget()
        self.paramStack.addWidget(self.lrPage)

        # fields
        self.lrRegularizationComboBox = QComboBox(self.lrPage)
        self.lrRegularizationComboBox.addItem('l2')
        self.lrRegularizationComboBox.addItem('l1')
        self.lrRegularizationListView = self.lrRegularizationComboBox.view()
        self.lrRglrStrengthLineEdit = QLineEdit('1.0', self.lrPage)
        self.lrFitInterceptCheckBox = QCheckBox('', self.lrPage)
        self.lrFitInterceptCheckBox.setChecked(True)
        self.lrInterceptScalingLineEdit = QLineEdit('1.0', self.lrPage)
        self.lrSolverComboBox = QComboBox(self.lrPage)
        self.lrSolverComboBox.addItem('liblinear')
        self.lrSolverComboBox.addItem('lbfgs')
        self.lrSolverComboBox.addItem('sag')
        self.lrSolverComboBox.addItem('saga')
        self.lrSolverComboBox.addItem('newton-cg')
        self.lrMultiClassComboBox = QComboBox(self.lrPage)
        self.lrMultiClassComboBox.addItem('ovr')
        self.lrMultiClassComboBox.addItem('multinomial')
        self.lrMultiClassListView = self.lrMultiClassComboBox.view()
        self.lrClassWeightComboBox = QComboBox(self.lrPage)
        self.lrClassWeightComboBox.addItem('uniform')
        self.lrClassWeightComboBox.addItem('balanced')

        self.lrStopLabel = QLabel('Stopping Criteria:', self.lrPage)
        self.lrTolLabel = QLabel('tol:', self.lrPage)
        self.lrTolLabel.setMinimumWidth(60)
        self.lrTolLineEdit = QLineEdit('1e-4', self.lrPage)
        self.lrMaxIterLabel = QLabel('max_iter:', self.lrPage)
        self.lrMaxIterLabel.setMinimumWidth(60)
        self.lrMaxIterLineEdit = QLineEdit('100', self.lrPage)

        # layout
        self.lrLayout = QFormLayout()
        self.lrLayout.addRow('penalty_type:', self.lrRegularizationComboBox)
        self.lrLayout.addRow('penalty:', self.lrRglrStrengthLineEdit)
        self.lrLayout.addRow('fit_intercept:', self.lrFitInterceptCheckBox)
        self.lrLayout.addRow('intercept_scaling:', self.lrInterceptScalingLineEdit)
        self.lrLayout.addRow('solver:', self.lrSolverComboBox)
        self.lrLayout.addRow('multi_class:', self.lrMultiClassComboBox)
        self.lrLayout.addRow('class_weight:', self.lrClassWeightComboBox)

        self.lrSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.lrStopLayout = QHBoxLayout()
        self.lrStopLayout.addWidget(self.lrTolLabel)
        self.lrStopLayout.addWidget(self.lrTolLineEdit)
        self.lrStopLayout.addWidget(self.lrMaxIterLabel)
        self.lrStopLayout.addWidget(self.lrMaxIterLineEdit)

        self.lrPageLayout = QVBoxLayout(self.lrPage)
        self.lrPageLayout.addLayout(self.lrLayout)
        self.lrPageLayout.addItem(self.lrSpacer)
        self.lrPageLayout.addWidget(self.lrStopLabel)
        self.lrPageLayout.addLayout(self.lrStopLayout)

        ## Neural Network
        self.nnIcon = QIcon('icons/nn.png')
        self.classTypeComboBox.addItem(self.nnIcon, 'Neural Network')
        self.nnPage = QWidget()
        self.paramStack.addWidget(self.nnPage)

        # fields
        self.nnNumHiddenUnitsSpinBox = self.spin_box(100, 1, 200, 1, self.nnPage)
        self.nnActivationComboBox = QComboBox(self.nnPage)
        self.nnActivationComboBox.addItem('relu')
        self.nnActivationComboBox.addItem('logistic')
        self.nnActivationComboBox.addItem('tanh')
        self.nnActivationComboBox.addItem('identity')
        self.nnSolverComboBox = QComboBox(self.nnPage)
        self.nnSolverComboBox.addItem('adam')
        self.nnSolverComboBox.addItem('lbfgs')
        self.nnSolverComboBox.addItem('sgd')
        self.nnAlphaLineEdit = QLineEdit('1e-4', self.nnPage)
        self.nnBatchSizeLineEdit = QLineEdit('20', self.nnPage)
        self.nnLearningRateComboBox = QComboBox(self.nnPage)
        self.nnLearningRateComboBox.addItem('constant')
        self.nnLearningRateComboBox.addItem('invscaling')
        self.nnLearningRateComboBox.addItem('adaptive')
        self.nnLearningRateInitLineEdit = QLineEdit('0.001', self.nnPage)
        self.nnEarlyStoppingCheckBox = QCheckBox('', self.nnPage)

        self.nnStopLabel = QLabel('Stopping Criteria:', self.nnPage)
        self.nnTolLabel = QLabel('tol:', self.nnPage)
        self.nnTolLabel.setMinimumWidth(60)
        self.nnTolLineEdit = QLineEdit('1e-3', self.nnPage)
        self.nnMaxIterLabel = QLabel('max_iter:', self.nnPage)
        self.nnMaxIterLabel.setMinimumWidth(60)
        self.nnMaxIterLineEdit = QLineEdit('200', self.nnPage)

        # layout
        self.nnLayout = QFormLayout()
        self.nnLayout.addRow('num_hidden_units:', self.nnNumHiddenUnitsSpinBox)
        self.nnLayout.addRow('activation:', self.nnActivationComboBox)
        self.nnLayout.addRow('solver:', self.nnSolverComboBox)
        self.nnLayout.addRow('penalty:', self.nnAlphaLineEdit)
        self.nnLayout.addRow('batch_size:', self.nnBatchSizeLineEdit)
        self.nnLayout.addRow('learning_rate:', self.nnLearningRateComboBox)
        self.nnLayout.addRow('learning_rate_init:', self.nnLearningRateInitLineEdit)
        self.nnLayout.addRow('early_stopping:', self.nnEarlyStoppingCheckBox)

        self.nnSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.nnStopLayout = QHBoxLayout()
        self.nnStopLayout.addWidget(self.nnTolLabel)
        self.nnStopLayout.addWidget(self.nnTolLineEdit)
        self.nnStopLayout.addWidget(self.nnMaxIterLabel)
        self.nnStopLayout.addWidget(self.nnMaxIterLineEdit)

        self.nnPageLayout = QVBoxLayout(self.nnPage)
        self.nnPageLayout.addLayout(self.nnLayout)
        self.nnPageLayout.addItem(self.nnSpacer)
        self.nnPageLayout.addWidget(self.nnStopLabel)
        self.nnPageLayout.addLayout(self.nnStopLayout)

        ## SVM
        self.svmIcon = QIcon('icons/svm.png')
        self.classTypeComboBox.addItem(self.svmIcon, 'SVM')
        self.svmPage = QWidget()
        self.paramStack.addWidget(self.svmPage)

        # fields
        self.svmPenaltyLineEdit = QLineEdit('1.0', self.svmPage)
        self.svmKernelComboBox = QComboBox(self.svmPage)
        self.svmKernelComboBox.addItem('rbf')
        self.svmKernelComboBox.addItem('linear')
        self.svmKernelComboBox.addItem('poly')
        self.svmKernelComboBox.addItem('sigmoid')
        self.svmDegreeSpinBox = self.spin_box(3, 1, 5, 1, self.svmPage)
        self.svmDegreeSpinBox.setDisabled(True)
        self.svmGammaLineEdit = QLineEdit('', self.svmPage)
        self.svmCoefLineEdit = QLineEdit('0.0', self.svmPage)
        self.svmClassWeightComboBox = QComboBox(self.svmPage)
        self.svmClassWeightComboBox.addItem('uniform')
        self.svmClassWeightComboBox.addItem('balanced')

        self.svmStopLabel = QLabel('Stopping Criteria:', self.svmPage)
        self.svmTolLabel = QLabel('tol:', self.svmPage)
        self.svmTolLabel.setMinimumWidth(60)
        self.svmTolLineEdit = QLineEdit('1e-3', self.svmPage)
        self.svmMaxIterLabel = QLabel('max_iter:', self.svmPage)
        self.svmMaxIterLabel.setMinimumWidth(60)
        self.svmMaxIterLineEdit = QLineEdit('200', self.svmPage)

        # layout
        self.svmLayout = QFormLayout()
        self.svmLayout.addRow('penalty:', self.svmPenaltyLineEdit)
        self.svmLayout.addRow('kernel:', self.svmKernelComboBox)
        self.svmLayout.addRow('degree:', self.svmDegreeSpinBox)
        self.svmLayout.addRow('kernel_coef:', self.svmGammaLineEdit)
        self.svmLayout.addRow('indenpendent_term:', self.svmCoefLineEdit)
        self.svmLayout.addRow('class_weight:', self.svmClassWeightComboBox)

        self.svmSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.svmStopLayout = QHBoxLayout()
        self.svmStopLayout.addWidget(self.svmTolLabel)
        self.svmStopLayout.addWidget(self.svmTolLineEdit)
        self.svmStopLayout.addWidget(self.svmMaxIterLabel)
        self.svmStopLayout.addWidget(self.svmMaxIterLineEdit)

        self.svmPageLayout = QVBoxLayout(self.svmPage)
        self.svmPageLayout.addLayout(self.svmLayout)
        self.svmPageLayout.addItem(self.svmSpacer)
        self.svmPageLayout.addWidget(self.svmStopLabel)
        self.svmPageLayout.addLayout(self.svmStopLayout)

        ## Naive bayes
        self.nbIcon = QIcon('icons/nb.png')
        self.classTypeComboBox.addItem(self.nbIcon, 'Naive Bayes')
        self.nbPage = QWidget()
        self.paramStack.addWidget(self.nbPage)

        # fields
        self.nbDistributionLabel = QLabel('')
        self.nbAddSmoothDoubleSpinBox = self.double_spin_box(1, 0, 50, 0.5, 1, self.nbPage)
        self.nbFitPriorCheckBox = QCheckBox(self.nbPage)
        self.nbFitPriorCheckBox.setChecked(True)
        self.nbClassPriorLineEdit = QLineEdit('None', self.nbPage)

        # layout
        self.nbLayout = QFormLayout()
        self.nbLayout.addRow('distributon:', self.nbDistributionLabel)
        self.nbLayout.addRow('smoothing:', self.nbAddSmoothDoubleSpinBox)
        self.nbLayout.addRow('fit_prior:', self.nbFitPriorCheckBox)
        self.nbLayout.addRow('class_prior:', self.nbClassPriorLineEdit)

        self.nbSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.nbPageLayout = QVBoxLayout(self.nbPage)
        self.nbPageLayout.addLayout(self.nbLayout)
        self.nbPageLayout.addItem(self.nbSpacer)

        ### classifier name
        self.classNameLabel = self.title('Classifier Name:', self.modelPage)
        self.classNameLineEdit = QLineEdit(self.modelPage)
        self.classNameLineEdit.setDisabled(True)
        self.classNameLayout = QHBoxLayout()
        self.classNameLayout.addWidget(self.classNameLabel)
        self.classNameLayout.addWidget(self.classNameLineEdit)
        self.modelPageLayout.addLayout(self.classNameLayout)

        ### comment
        self.classCommentLabel = self.title('Comment (no more than 30 characters):', self.modelPage)
        self.classCommentTextEdit = QTextEdit(self.modelPage)
        self.classCommentTextEdit.setMaximumHeight(50)
        self.classCommentTextEdit.setDisabled(True)

        self.modelPageLayout.addWidget(self.classCommentLabel)
        self.modelPageLayout.addWidget(self.classCommentTextEdit)

        ### reset and train buttons
        self.classResetTrainSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.classResetPushButton = QPushButton('Reset', self.modelPage)
        self.classResetPushButton.setMinimumWidth(90)
        self.classResetPushButton.setDisabled(True)
        self.classTrainPushButton = QPushButton('Train', self.modelPage)
        self.classTrainPushButton.setMinimumWidth(90)
        self.classTrainPushButton.setDefault(True)
        self.classTrainPushButton.setDisabled(True)

        self.classResetTrainLayout = QHBoxLayout()
        self.classResetTrainLayout.addItem(self.classResetTrainSpacer)
        self.classResetTrainLayout.addWidget(self.classResetPushButton)
        self.classResetTrainLayout.addWidget(self.classTrainPushButton)

        self.modelPageLayout.addLayout(self.classResetTrainLayout)

        ### page spacer and line
        self.modelPageSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.modelPageLayout.addItem(self.modelPageSpacer)
        self.modelLine = QFrame(self.modelPage)
        self.modelLine.setFrameShape(QFrame.HLine)
        self.modelLine.setFrameShadow(QFrame.Sunken)
        self.modelPageLayout.addWidget(self.modelLine)

        ### back and next buttons
        self.classBackNextSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.classBackPushButton = QPushButton('Back', self.modelPage)
        self.classBackPushButton.setMinimumWidth(90)
        self.classNextPushButton = QPushButton('Next', self.modelPage)
        self.classNextPushButton.setMinimumWidth(90)
        self.classNextPushButton.setDefault(True)
        self.classNextPushButton.setDisabled(True)
        self.classBackNextLayout = QHBoxLayout()
        self.classBackNextLayout.addItem(self.classBackNextSpacer)
        self.classBackNextLayout.addWidget(self.classBackPushButton)
        self.classBackNextLayout.addWidget(self.classNextPushButton)

        self.modelPageLayout.addLayout(self.classBackNextLayout)

        self.leftPanel.addWidget(self.modelPage)

        # signal handling
        self.classTypeComboBox.currentIndexChanged.connect(self.paramStack.setCurrentIndex)
        self.lrSolverComboBox.currentIndexChanged.connect(self.update_logistic_regression)
        self.nnSolverComboBox.currentIndexChanged.connect(self.update_neural_network)
        self.svmKernelComboBox.currentIndexChanged.connect(self.update_svm)

        self.paramStack.currentChanged.connect(\
            lambda: self.classNameLineEdit.setEnabled(self.paramStack.currentIndex() > 0))
        self.paramStack.currentChanged.connect(\
            lambda: self.classCommentTextEdit.setEnabled(self.paramStack.currentIndex() > 0))
        self.paramStack.currentChanged.connect(\
            lambda: self.classTrainPushButton.setEnabled(self.paramStack.currentIndex() > 0))
        self.paramStack.currentChanged.connect(\
            lambda: self.classResetPushButton.setEnabled(self.paramStack.currentIndex() > 0))

        self.classTrainPushButton.clicked.connect(self.train)
        self.classResetPushButton.clicked.connect(self.reset)

        self.classBackPushButton.clicked.connect(lambda: self.clear(option='train'))
        self.classNextPushButton.clicked.connect(lambda: self.leftPanel.setCurrentIndex(2))

        ########## 3rd page: model selection and testing ##########
        self.testPage = QWidget()
        self.testPageLayout = QVBoxLayout(self.testPage)

        ### classifier selection
        self.classSelectLabel = self.title('Classifier Selection:', self.testPage)
        self.testPageLayout.addWidget(self.classSelectLabel)

        self.classSelectFrame = QFrame(self.testPage)
        self.classSelectFrame.setAutoFillBackground(True)
        self.classSelectLayout = QVBoxLayout(self.classSelectFrame)
        self.bestPerformRadioButton = QRadioButton('Best-Performing', self.classSelectFrame)
        self.bestPerformRadioButton.setChecked(True)
        self.userPickRadioButton = QRadioButton('User-Picked', self.classSelectFrame)
        self.bestPerformSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.metricLabel = QLabel('Metric:', self.classSelectFrame)
        self.metricComboBox = QComboBox(self.classSelectFrame)
        self.metricComboBox.addItem('--- select ---')
        self.metricComboBox.addItem('accuracy')
        self.metricComboBox.addItem('precision')
        self.metricComboBox.addItem('recall')
        self.metricComboBox.addItem('f1')
        self.metricComboBox.addItem('AUROC')
        self.metricComboBox.addItem('AUPRC')
        self.userPickLabel = QLabel('None', self.classSelectFrame)
        self.userPickLabel.setMinimumWidth(150)

        self.bestPerformLayout = QHBoxLayout()
        self.bestPerformLayout.addWidget(self.bestPerformRadioButton)
        self.bestPerformLayout.addItem(self.bestPerformSpacer)
        self.bestPerformLayout.addWidget(self.metricLabel)
        self.bestPerformLayout.addWidget(self.metricComboBox)
        self.classSelectLayout.addLayout(self.bestPerformLayout)

        self.userPickLayout = QHBoxLayout()
        self.userPickLayout.addWidget(self.userPickRadioButton)
        self.userPickLayout.addWidget(self.userPickLabel)
        self.classSelectLayout.addLayout(self.userPickLayout)

        self.testPageLayout.addWidget(self.classSelectFrame)

        ### test button
        self.testPushButton = QPushButton('Test')
        self.testPushButton.setMinimumWidth(90)
        self.testPushButton.setDefault(True)
        self.testPushButton.setDisabled(True)
        self.testSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.testLayout = QHBoxLayout()
        self.testLayout.addItem(self.testSpacer)
        self.testLayout.addWidget(self.testPushButton)
        self.testPageLayout.addLayout(self.testLayout)

        ### prediction button
        self.predictionLabel = self.title('Prediction:', self.testPage)
        self.predictionPushButton = QPushButton('Predict and Save as...', self.testPage)
        self.predictionPushButton.setMaximumWidth(175)
        self.predictionPushButton.setDisabled(True)
        self.predictionLayout = QHBoxLayout()
        self.predictionLayout.addWidget(self.predictionLabel)
        self.predictionLayout.addWidget(self.predictionPushButton)
        self.testPageLayout.addLayout(self.predictionLayout)

        ### page spacer and line
        self.testPageSpacer = QSpacerItem(40, 20,QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.testPageLayout.addItem(self.testPageSpacer)
        self.testLine = QFrame(self.testPage)
        self.testLine.setFrameShape(QFrame.HLine)
        self.testLine.setFrameShadow(QFrame.Sunken)
        self.testPageLayout.addWidget(self.testLine)

        ### back and finish buttons
        self.testBackPushButton = QPushButton('Back', self.testPage)
        self.testBackPushButton.setMinimumWidth(90)
        self.testFinishPushButton = QPushButton('Finish', self.testPage)
        self.testFinishPushButton.setMinimumWidth(90)
        self.testFinishPushButton.setDefault(True)
        self.testFinishPushButton.setDisabled(True)        
        self.testBackFinishSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.testBackFinishLayout = QHBoxLayout()
        self.testBackFinishLayout.addItem(self.testBackFinishSpacer)
        self.testBackFinishLayout.addWidget(self.testBackPushButton)
        self.testBackFinishLayout.addWidget(self.testFinishPushButton)
        self.testPageLayout.addLayout(self.testBackFinishLayout)

        self.leftPanel.addWidget(self.testPage)

        # signal handling
        self.bestPerformRadioButton.toggled.connect(self.metricComboBox.setEnabled)
        self.bestPerformRadioButton.toggled.connect(lambda: self.select(option='best'))
        self.metricComboBox.currentIndexChanged.connect(lambda: self.select(option='best'))

        self.userPickRadioButton.toggled.connect(self.metricComboBox.setDisabled)
        self.userPickRadioButton.toggled.connect(lambda: self.select(option='user'))
        
        self.testPushButton.clicked.connect(self.test)
        self.predictionPushButton.clicked.connect(self.predict)

        self.testBackPushButton.clicked.connect(lambda: self.clear(option='test'))
        self.testFinishPushButton.clicked.connect(self.finish)

        ########## right panel ##########
        self.rightPanelLayout = QVBoxLayout(self.rightPanel)

        self.trainedClassifiersLayout = QHBoxLayout()
        self.trainedClassifiersLabel = self.title('Trained Classifiers:', self.rightPanel)
        self.performanceLabel = QLabel('Show performance on ', self.rightPanel)
        self.performanceComboBox = QComboBox(self.rightPanel)
        self.performanceListView = self.performanceComboBox.view()
        self.performanceComboBox.addItem('validation data')
        self.performanceComboBox.addItem('training data')
        self.performanceComboBox.addItem('test data')
        self.performanceListView.setRowHidden(2, True)

        self.trainedClassifiersSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.trainedClassifiersLayout.addWidget(self.trainedClassifiersLabel)
        self.trainedClassifiersLayout.addItem(self.trainedClassifiersSpacer)
        self.trainedClassifiersLayout.addWidget(self.performanceLabel)
        self.trainedClassifiersLayout.addWidget(self.performanceComboBox)
        self.models_table = QTableWidget(self.rightPanel)
        self.models_table.verticalHeader().setDefaultSectionSize(15)
        self.models_table.verticalHeader().hide()
        self.models_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.models_table.setColumnCount(8)
        self.models_table.setHorizontalHeaderLabels(['Name', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC'])
        self.models_table.setColumnWidth(0, 140)
        self.models_table.setColumnWidth(1, 110)
        self.models_table.setColumnWidth(2, 60)
        self.models_table.setColumnWidth(3, 60)
        self.models_table.setColumnWidth(4, 60)
        self.models_table.setColumnWidth(5, 60)
        self.models_table.setColumnWidth(6, 60)
        self.models_table.setColumnWidth(7, 60)
        self.models_table.setFont(self.font)
        self.table_header = self.models_table.horizontalHeader()
        self.rightPanelLayout.addLayout(self.trainedClassifiersLayout)
        self.rightPanelLayout.addWidget(self.models_table)

        self.visLayout = QHBoxLayout()
        self.visListLayout = QVBoxLayout()
        self.model_summary_ = QTreeWidget(self.rightPanel)
        self.model_summary_.setMaximumWidth(260)
        self.model_summary_.setColumnCount(2)
        self.model_summary_.setHeaderHidden(True)
        self.model_summary_.setColumnWidth(0, 150)
        self.model_summary_.setFont(self.font)
        self.visListLayout.addWidget(self.model_summary_)

        self.visFrame = QFrame(self.rightPanel)
        self.visFrameLayout = QHBoxLayout(self.visFrame)
        self.dataPlotRadioButton = QRadioButton('Data Plot', self.visFrame)
        self.dataPlotRadioButton.setDisabled(True)
        self.rocRadioButton = QRadioButton('ROC', self.visFrame)
        self.confusionMatrixRadioButton = QRadioButton('Confusion Matrix', self.visFrame)
        self.confusionMatrixRadioButton.setChecked(True)
        self.prRadioButton = QRadioButton('Precision-Recall', self.visFrame)
        
        self.visFrameLeftLayout = QVBoxLayout()
        self.visFrameLeftLayout.addWidget(self.dataPlotRadioButton)
        self.visFrameLeftLayout.addWidget(self.rocRadioButton)
        self.visFrameRightLayout = QVBoxLayout()
        self.visFrameRightLayout.addWidget(self.confusionMatrixRadioButton)
        self.visFrameRightLayout.addWidget(self.prRadioButton)
        self.visFrameLayout.addLayout(self.visFrameLeftLayout)
        self.visFrameLayout.addLayout(self.visFrameRightLayout)
        self.visListLayout.addWidget(self.visFrame)
        self.visLayout.addLayout(self.visListLayout)

        self.canvas = FigureCanvas(Figure(figsize=(340, 340)))
        self.canvas.setMaximumWidth(340)
        self.canvas.setMaximumHeight(340)
        self.canvas.setParent(self.rightPanel)
        self.visLayout.addWidget(self.canvas)

        self.rightPanelLayout.addLayout(self.visLayout)

        # signal handling
        self.performanceComboBox.currentIndexChanged.connect(self.switch_metrics)
        self.performanceComboBox.currentIndexChanged.connect(\
            lambda: self.bestPerformRadioButton.setDisabled(self.performanceComboBox.currentIndex() == 2))
        self.performanceComboBox.currentIndexChanged.connect(\
            lambda: self.metricComboBox.setDisabled(self.performanceComboBox.currentIndex() == 2 \
                or not self.bestPerformRadioButton.isChecked()))
        self.performanceComboBox.currentIndexChanged.connect(\
            lambda: self.userPickRadioButton.setDisabled(self.performanceComboBox.currentIndex() == 2))
        self.performanceComboBox.currentIndexChanged.connect(\
            lambda: self.testPushButton.setDisabled(self.performanceComboBox.currentIndex() == 2 \
                or self.selected_model == None))
        self.performanceComboBox.currentIndexChanged.connect(\
            lambda: self.table_header.setSectionsClickable(self.performanceComboBox.currentIndex() != 2))
        self.models_table.cellClicked.connect(self.switch_model)
        self.table_header.sectionClicked.connect(self.sort)

        self.confusionMatrixRadioButton.toggled.connect(self.plot)
        self.rocRadioButton.toggled.connect(self.plot)
        self.prRadioButton.toggled.connect(self.plot)

        # global geometry
        self.resize(1024, 700)
        self.setWindowTitle('ML4Bio')
        self.leftPanel.resize(360, 680)
        self.leftPanel.setStyleSheet("QStackedWidget {background-color:rgb(226, 226, 226)}")
        self.leftPanel.move(10, 10)
        self.rightPanel.resize(640, 680)
        self.rightPanel.move(380, 10)

        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())