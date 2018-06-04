# Machine Learning for Biologists Workshop

Recent advances in high-throughput technologies have led to rapid growth in the amount of omics data. These massive datasets, accompanied by numerous data points enriched over the past decades, are extremely valuable in the discovery of complex biological structures, but also present big challenges in various aspects of data analysis. This workshop introduces biologists to machine learning, a powerful set of models for drawing inference from big data.  Upon completion of the workshop, attendees will gain expertise in building simple predictive models and become more comfortable to collaborate with computational experts on more complicated data analysis tasks.

## Tentative sections:

### Machine learning workflow

**Overview**
  * Task definition: classification (main focus), regression, clustering, etc.
  * Data representation: 
    * Features: categorical, continuous
    * Labels: one endpoint (main focus), multi-task
    * File format: txt, json, csv, etc.
    * Input (feature vectors) -> model (functions) -> output (predictions)
    
**Pipeline**
  * Data preparation: cleaning, integration, etc.
  * Iteratively:
    * Feature engineering (deserves a stand-alone section)
    * Model building: training, validation, evaluation
    * Model selection
    * Feature selection
  * Testing
  * Interpretation

### Introduction to Classification models

Implement a GUI.
Expose key hyperparameters to user.
Show initial model state including model structure and initial parameters.
(Optional) Show updated model state after every few epochs.
Show decision boundary (when applicable).

**Decision trees**

**Random forests**

**K-nearest neighbors**

* Unweighted and weighted versions (as hyperparameter)
* Use toy data to show 2D or 3D case.
* In the unweighted version, use different colors to distinguish between effective and ineffective data points.
* In the weighted version, visualize distance through color change (gray scale: the closer, the darker).
* Curse of dimensionality?

**Logistic Regression**

**Neural Networks**

**Support Vector Machines**

**Naive Bayes**

More to be added... 

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
