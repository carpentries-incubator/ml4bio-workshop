# Machine Learning (ML) for Biologists Workshop

Recent advances in high-throughput technologies have led to rapid growth in the amount of omics data. These massive datasets, accompanied by numerous data points enriched over the past decades, are extremely valuable in the discovery of complex biological structures, but also present big challenges in various aspects of data analysis. This workshop introduces biologists to machine learning, a powerful set of models for drawing inference from big data.  Upon completion of the workshop, attendees will gain expertise in building simple predictive models and become more comfortable to collaborate with computational experts on more complicated data analysis tasks.

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

### ML models for classification

* A GUI (developed using Kivy or PyQt framework) with two modes:
  * Teaching mode with built-in toy datasets (2D input) and data/model visualization 
    (reference: http://playground.tensorflow.org)
  * Production mode with user-supplied datasets and without visualization 
    (reference: https://github.com/gmiaslab/ClassificaIO)
* Models to be included:
  * Decision trees (viewed as a special case of RF)
  * Random forests
  * K-nearest neighbors
  * Logistic regression (viewed as a special case of NN)
  * Neural networks
  * Support vector machines
  * Naive bayes 

### Training, validation, testing and evaluation

Expand on the following topics:
* Training set, validation set, test set
* Stopping criteria
* K-fold cross-validation
* Leave-one-out
* Evaluation metrics: accuracy (error rate), precision, recall, F1, etc.
* Graphics: ROC, Precision-recall curve, confusion matrix

Unique issues in biological application:
* How to obtain negative examples?
* Unbalanced data
* Cost-sensitive analysis (choosing threshold)

### Model selection

Expand on the following topics:
* Overfitting
* Outliers

### Feature selection

### Model interpretation

### Feature engineering


to be continued...
