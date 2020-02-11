## Guideline

### Fundamentals of ML

* Task definition: classification (main focus), regression, clustering, etc.
* Data representation:
  * Features: categorical, numerical (discrete/continuous)
  * Labels: univariate (binary/multi-class), multivariate
  * File formats: txt, json, csv, etc.
  * Input (feature vectors) -> model (functions) -> output (predictions)
  * Learning: parameter estimation
  * Prediction: function evaluation

### ML in Biological Data Analysis: Pipeline

* Problem formulation
* Data preparation: cleaning, integration, etc.
* Iteratively:
  * Feature engineering
  * Model building: training, validation, evaluation
  * Model selection
  * Feature selection
* Testing
* Interpretation

### Models for Classification: An Introduction

* A GUI (developed using PyQt5) with two modes:
  * Playground: built-in toy datasets, data/model visualization
    (reference: http://playground.tensorflow.org)
  * Production: user-supplied datasets, no visualization
    (reference: https://github.com/gmiaslab/ClassificaIO)
* Models to be included:
  * Decision trees (viewed as a special case of RF)
  * Random forests
  * K-nearest neighbors
  * Logistic regression (viewed as a special case of NN)
  * Neural networks
  * Support vector machines
  * Naive bayes

### Models for Classification: Training and Evaluation

Expand on the following topics:
* Training set, validation set, test set
* Stopping criteria
* K-fold cross-validation
* Leave-one-out
* Evaluation metrics: accuracy (error rate), precision, recall, F1, etc.
* Graphics: ROC, Precision-recall curve, confusion matrix

Unique issues in biological application:
* Small size of data
* How to obtain negative examples?
* Unbalanced data
* Sparsity
* Cost-sensitive analysis (How to pick threshold?)

### Model Selection

Expand on the following topics:
* Overfitting
* Outliers

### Feature selection

### Model interpretation

### Feature engineering


to be continued...
