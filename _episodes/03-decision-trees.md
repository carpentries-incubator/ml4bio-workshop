---
title: "Decision Trees"
teaching: 15
questions:
- "What is the decision rule?"
objectives:
- "Compare and contrast the strengths and weaknesses of machine learning classifiers commonly used in biology - logistic regression, decision trees, random forests, and neural networks. Assess model selection and recognize that these methods don't necessarily work right out of the box."
- "Discuss applying, updating, and evaluating a trained model on new data."
---
_Comment: Is the objective to learn about pros/cons of the decision tree model as opposed to all of these models?_
_Comment: I'm not sure about the decision rule as a central question_

### What is the decision tree classifier? 

Decision trees make predictions by asking a sequence of questions for each example and make a prediction based on the responses.
This makes decision trees intuitive.
One of the benefits is that we can clearly see the path of questios and answers we took to get to the final prediction.
For example, a doctor might use a decision tree to decide which medicine to prescribe based on a patient's responses about their symptoms.
Or in the case of T-cells, a decision tree can predict whether or not a T cell is active or quiescent.

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decision%20tree1.jpg">
</p>

#### Example

To better understand the algorithm, let's consider the house price prediction example from the [Introduction episode](lhttps://gitter-lab.github.io/ml-bio-workshop/01-introduction/index.html).
We are going to begin with an initial house price range, and for the neighborhood of our interest the prices range from $100k - $250k. 
The first question we could ask is the number of bedrooms in the house. 
The answer is 3 bedrooms, and so our new range will be $180k-$250k. 
Then we will ask about the number of bathrooms, and the answer is 3 bathrooms. 
The new range range is $220-$250.
Finally we will ask 
That give us the price of $230k. 
_Comment: (add the visual)_

#### How does the classifier make predictions?

This intuitive way of understanding decision trees is very close to the way the algorithm is implemented, but we also have the other part of the split to consider. 
Each split of a decision tree classifies instances based on a test of a single feature. 
This test can be True or False. 
The splitting goes from the root at the top of the tree to a leaf node at the bottom. 
So an instance is classified starting from the root and testing the feature specified by the node, then going down the split based on the outcome of the test and testing a different feature specified by another node.  
Refer to the visual. 

_Comment - add a visual_

### About the classifier

A decision tree is a supervised learning classifier.
It splits the initial population depending on a certain rule.
The goal of the classifier is to predict the class label on a new set of data based on the rule that the classifier learned from the features of the training examples.
An important property of the decision tree is the depth of tree.
That is the number of times we make a split to reach a decition. 

Some pros of using decision trees:

- easy to visualize
- the classification can be visually followed, so it is easy to reproduce
- makes few assumptions about the data
- can ignore useless features

Some cons of using decision trees:

- prone to overfitting
- requires a way to turn numeric data into a single decision rule

### Step 1 Select data

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the decision tree look graphically?
{: .challenge}

Let's load *simulated_t_cells_7.csv* data set into the software.
This dataset is engineered specifically to demonstrate the decision tree classifier.

We will continue working on the T-cells example.
The goal is the same, predicting whether a cell is active or inactive.
We also have the same two features: cell size and intensity. 

### Step 2 Train classifiers

In the original T-cells example, we left the hyperparameters settings as default.
Now we will look further into some of the hyperparameters.
In this workshop, not all of the hyperparameters from the software will be covered.
For the hyperparameters that we don't discuss, use the default settings.
- Max_depth can be an integer or None. It is the maximum depth of the tree. If the max depth is set to None, the tree nodes are fully expanded or until they have less than min_samples_split samples.
- Min_samples_split and min_samples_leaf represent the minimum number of samples required to split a node or to be at a leaf node.
- Class_weight is important hyperparameter in biology research. If we had a training set and we are using binary classification, i.e., in the T-cells example, if 2 samples are labeled as active and 98 samples are labeled as quiescent, we don't want to train the model that predicts all of the cells to be quiescent. Class_weight parameter would allow putting weight on 2 cells labeled as active. 
In biology, it is common to have an imbalanced training set with more negative than positive instances, so training and evaluating appropriately is essential! The uniform mode gives all classes the weight one. The balanced mode adjusts the weights.

Without changing any hyperparameter settings, look at the Data Plot.

**What do you notice?**

The data plot shows two features, where the blue data points represent the quiescent cells, and the red data points represent the active cells. 

**What hyperparameter might be important for this example?**

The given dataset is imbalanced with more quiescent cells than active cells. 
Let's change the class_weight to balanced. 
Did this make any difference? 
How does the data plot look for the uniform class_weight and how does it look for the balanced class weight?


### Step 3 Test and predict

#### Overfitting
_Comment: It'd be tough to construct an example of decision trees overfitting if we stick to the toy_data examples with only 2 features. We may need to use another one of the datasets to show overfitting in this context._
If a model fits the training data perfectly, or very well, the model can become too data dependent and not work as well on a new data. 

_Comment: We need more callouts to guide this lesson.  What steps should they take?  What hyperparameter values should they try?  What are the key ideas?  What do they observe about the decision boundaries of a decision tree?_

> ## Key points
> 
> Decision trees are easy to visualize and intuitive to understand. 
> Decisition trees are prone to overfitting.
> In biology, imbalanced datasets are common..
{: .keypoints}

###  Application in biology

[PgpRules: a decision tree based prediction server for P-glycoprotein substrates and inhibitors](https://doi.org/10.1093/bioinformatics/btz213)
