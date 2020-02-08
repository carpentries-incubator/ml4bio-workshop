### Part I: Introduction (45min)
* Motivation (30min):
    * High-throughput data in biology
    * Why biologists should learn ML
    * Motivating examples of ML in biology
    * Logistics (15min):  
	  * Learning objectives  
	  * Schedule  
	  * Software setup `Tony: budget more time for this if it includes Anaconda setup`

### Part II: ML basics (3hr45min)
* ML vocabulary (30mins) `Tony: do we want to interweave the vocabulary with the workflow to make it more interactive?`
    * Data format: features, labels, continuous/categorical data, etc.
    * Data encoding: integer encoding, one-hot encoding
    * Task: Classification, regression, clustering, etc.
    * Training vs. prediction (highlight)
    * Assumptions (highlight)
* Overview of ML workflow (30mins)
    * Introduce the main steps at high level
    * Work through an example in the software (This also introduces the software to the participants)
* Classifiers: training, evaluation and prediction (theory + practice with toy data) (2hr)
    * Logistic regression and Neural Network
      * Model architecture: input layer, hidden layer, output
      * Introduce sigmoid function
      * Input to a unit: linear combination of the values output from connected units in the previous layer, bias term
      * Output from a unit: activation functions, how to decide which is the predicted class
      * Data encoding: why one-hot encoding of discrete features
    * Practice: participants experiment on training logistic regression and neural networks
    * How to evaluate a classifier
      * Train/test split: why need a test set
      * Validation: hold-out validation set, k-fold cross-validation, leave-one-out, when to use which `Tony: do we introduce hyperparameters and regularization here?`
      * Metrics and plots: accuracy, TP, FP, TN, FN, precision, recall, f1, ROC, precision-recall, confusion matrix
      * Model selection: what metric to look at under what circumstances
      * Bad practices: Use training error in place of validation/test error, overfitting
    * Practice: participants experiment on model selection
    * Decision tree and random forest
      * Model architecture: internal nodes, leaves, logic rules
      * How to avoid overfitting: max height, min number of items required for splitting
      * Benefit of random forest: sampling data and features, variance reduction
    * Practice: participants experiment on training and model selection
    * Support Vector Machine
      * Margin: hard and soft margin, support vectors
      * Kernel: linear -> nonlinear, overfitting
    * Practice: participants experiment on training and model selection
    * K-nearest neighbors (if time permits)
    * Naïve Bayes (if time permits)
* Model Interpretation and Prediction (15mins)
* When ML fails (30mins)
    * Bad practice in model building
    * Overfitting, outliers, data size, high-dimensional data, etc.
    * Hidden variables, unseen classes
    * Change in experimental conditions

### Part III: ML application (1hr30min)
* Participants pick real biological datasets and build classifiers on their own
* Participants discuss their results and share their thoughts in a discussion session
(total time: 6hr)
