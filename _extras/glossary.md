---
title: Glossary
---
{% include links.md %}

__Machine learning__ - set of methods that can automatically detect patterns in data and then use those patterns to make predictions on future data or perform other kinds of decision making under uncertainty.

__Class label__ - prediction output, where the prediction is a category instead of a number.

__Supervised learning__ - training a model from the labeled input data.

__Unsupervised learning__ - training a model from the unlabeled input data to find the patterns in the data.

__Algorithm__ - a relationship between input and output. It is a set of steps that takes an input and produces an output.

__Decision boundary__ - a region where all the patterns within the decision boundary belong to the same class. It divides the space that is represented by all the data points. Identifying the decision boundary can be used to classify the data points. The decision boundary does not have to be linear.

__Model__ - mathematical representation that generates predictions based on the input data.

__Sample__ - a specific observation in a dataset. For instance, in the T-cells example each T-cell is a sample. Also called instances or observations.

__Class__ - the part of a dataset that is being predicted. In the T-cells example a T-cell's state as active or quiescent is its class. Also called the target variable or label. 

__Training set__ - the training set is a part of the original dataset that trains or fits the model. This is the data that the model uses to learn patterns.

__Validation set__ - a part of the training set is used to validate that the fitted model works on new data. This is not the final evaluation of the model. This step is used to change hyperparameters and then train the model again.

__Test set__ - the test set checks how well we expect the model to work on new data in the future. The test set is used in the final phase of the workflow, and it evaluates the final model. It can only be used one time, and the model cannot be adjusted after using it.

__Hyperparameters__ - these are the settings of a machine learning model. Each machine learning method has different hyperparameters, and they control various trade-offs which change how the model learns. Hyperparameters control parts of a machine learning method such as how much emphasis the method should place on being perfectly correct versus becoming overly complex, how fast the method should learn, the type of mathematical model the method should use for learning, and more.

__Classification__ - the task in supervised learning when the label is a category. The goal of classification is to predict which category each sample belongs to.

__Classifier__ - a specific model or algorithm which performs classification.

__Regression__ - the task in supervised learning when the label is numeric. Instead of predicting a category, here the value of the label variable is predicted.

__Confusion matrix__ - a matrix used in classification to visualize the performance of a classifier. Each cell shows the number of time the predicted and actual classes of samples occurred in a certain combination.

__Overfitting__ - an overfitting model fits the training data too well, but it fails to do this on the new data.

__Root node__ - the topmost node where the decision flow starts.

__Leaf node__ - a bottom node that doesn't split any further. It represents the class label.

__Node__ - a test performed on a feature. It branches into two branches.

__Imbalanced training set__ - a data set that contains a large proportion of a certain class or classes. 

__Depth of tree__ - the number of times we make a split to reach a decision.

__Evaluation metrics__ - used to measure the performance of a model.

__Confusion matrix__ - a table that summarizes successes and failures of model's predictions.

__Odds__ - probability that an event happens divided by probability that an event doesn't happen.

__Feature weights__ - determine the importance of a feature in a model.

__Linear separability__ - drawing a line in the plane that separates all the points of one kind on one side of the line, and all the point of the other kind on the other side of the line.
