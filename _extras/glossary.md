---
title: Glossary
---
{% include links.md %}

__Machine learning__ - set of methods that can automatically detect patterns in data and then use those patterns to make predictions on future data or perform other kinds of decision making under uncertainty.

__Class label__ - prediction output.

__Supervised learning__ - training a model from the labeled input data.

__Unsupervised learning__ - training a model from the unlabeled input data to find the patterns in the data.

__Algorithm__ - is a relationship between input and output. It is a set of steps that takes an input and produces an output.

__Model__ - mathematical representation that generates predictions based on the input data.

__Sample__ - A specific observation in a dataset. For instance, in the T-cells example each T-cell is a sample. Also called instances or observations.

__Class__ - The part of a dataset that is being predicted. In the T-cells example a T-cell's state as active or quiescent is its class. Also called the target variable or label. 

__Training set__ - The training set is a part of the original dataset that trains or fits the model. This is the data that the model uses to learn patterns.

__Validation set__ - Part of the training set is used to validate that the fitted model works on new data. This is not the final evaluation of the model. This step is used to change hyperparameters and then train the model again.

__Test set__ - The test set checks how well we expect the model to work on new data in the future. The test set is used in the final phase of the workflow, and it evaluates the final model. It can only be used one time, and the model cannot be adjusted after using it.

__Hyperparameters__ - These are the settings of a machine learning model. Each machine learning method has different hyperparameters, and they control various trade-offs which change how the model learns. Hyperparameters control parts of a machine learning method such as how much emphasis the method should place on being perfectly correct versus becoming overly complex, how fast the method should learn, the type of mathematical model the method should use for learning, and more.

__Classification__ - The task in supervised learning when the label is a category. The goal of classification is to predict which category each sample belongs to.

__Classifier__ - A specific model or algorithm which performs classification.

__Regression__ - The task in supervised learning when the label is numeric. Instead of predicting a category, here the value of the label variable is predicted.

__Confusion matrix__ - A matrix used in classification to visualize the performance of a classifier. Each cell shows the number of time the predicted and actual classes of samples occurred in a certain combination.

__Overfitting__ - an overfitting model was trained too well on the training data set. This might cause for the model not to work as well on the test data. 

__Root node__ - the topmost node where the decision flow starts.

__Leaf node__ - a bottom node that doesn't split any further. It represents the class label.

__Depth of tree__ - the number of times we make a split to reach a decision.

__Evaluation metrics__ - used to measure the performance of a model.

__Confusion matrix__ - a table that summarizes successes and failures of model's predictions.

__Odds__ - probability that an event happens divided by probability that an event doesn't happen.

__Linear Separability__ - drawing a line in the plane that separates all the points of one kind on one side of the line, and all the point of the other kind on the other side of the line.
