---
title: "Random Forests"
teaching: 30
questions:
- ""
objectives:
- "Discuss applying, updating, and evaluating a trained model on new data."
- "Select and evaluate a model on a sample dataset. Understand the models' complexity and limitations."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

_Comment: Need to add Overview info_
_Milica's comment: Review below_
In this episode, we will learn about random forests classifier. 
Using the software, we will analyze the pre-engineered dataset and discuss the hyperparameters. 
Then we will consider some applications in biology. 

We previously talked about how we would predict the price of a house using a decision tree. 
However, the prediction might not be accurate because many factors can affect it. 
To improve the prediction accuracy, we will talk to a thousand individuals who will use a decision tree classifier to make their prediction. 
All of these decision trees make a random forest. 
Random forest is a classifier that combines a large number of decision trees.
Each decision tree in the forest makes a prediction, and the final prediction is the class label that most decision trees predicted.
Combining many different decision trees makes the final prediction more robust and less prone to overfitting.

### Why is it called "random" forest?

If we give the same data to a thousand people, then they ask the same questions for each decision tree, and are provided with the same answers, we will get the same predictions that are prone to overfitting. 
In order to get the most accurate predictions we want to provide slightly different data to each person. 
When a human is making a decision, for example a wheather forecast in Madison, Wisconsin, if it is January, we know that it is winter and that it means the temperature will be colder. 
A classifier cannot rationalize this way. 
We need to provide a data that is a little bit different for each decision tree.
Then each decision tree will use a different subset of features to form the questions.
The feature selection will be random. 
The final prediction is based on the average taken across all the decision trees in the forest.

_Comment: Say more about the RF algortihm and how it randomlly selects features and instances to train on_
Milica's comment: please review

### Step 1 Select data

Let's load a T-cells #3 data set into our software. 
This dataset is designed to specifically illustrate properties of the random forest classifier.

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the random forest look graphically?
{: .challenge}

### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered.
For the hyperparameters that we don't discuss, use the default settings. 
- N-estimator represents the number of the decision trees that go into forest. Although we want to consider the biggest number of trees possible, there is a certain number where the classifier performance won't be improving. 
- Max-features represents the number of features considered. For the classification problems *sqrt* is mostly used. That is the square root of the total number of features.
- Bootstrap 


### Step 3 Test and Predict

**Finish once the dataset is done**

> ## Key points
>
> What classifier did we use, and why did we decide to use that one?
{: .keypoints}
