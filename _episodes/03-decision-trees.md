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

Decision trees make predictions by asking a sequence of questions for each example and make a prediction based on the responses.
This makes decision trees intuitive.
One of the benefits is that we can clearly see the path of questiosn and answers we took to get to the final prediction.
For example, a doctor might use a decision tree to decide which medicine to prescribe based on a patient's responses about their symptoms.
Or in the case of T-cells, a decision tree can predict whether or not a T cell is active or inactive.

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decision%20tree1.jpg">
</p>

To better understand the algorithm, let's consider a real-life example. 
_Comment: (add the visual)_

### Decision tree algorithm

A decision tree is a supervised learning classifier.
It splits the initial population depending on a certain rule.
The goal of the classifier is to predict the class label on a new set of data based on the rule that the classifier learned from the features of the training examples.
An important property of the decision tree is the depth of tree.
That is the number of times we make a split to reach a decition. 

Some pros of using decision trees:

- easy to visualize
- the classification can be visually followed, so it is easy to reproduce

Some cons of using decision trees:

- prone to overfitting

### Step 1 Select data

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the decision tree look graphically?
{: .challenge}

Let's load a T-cells #2 data set into the software. _Comment: What is the specific filename?_
This dataset is engineered specifically to demonstrate the decision tree classifier.
Please note that the prediction matrix for this dataset would rarely reflect the real-world data. _Comment: What is prediction matrix here?  Confusion matrix?_

We will continue working on the T-cells example.
The goal is the same, predicting whether a cell is active or inactive.
We also have the same two features: cell size and intensity. 


### Step 2 Train classifiers

In the original T-cells example, we left the hyperparameters settings as default.
Now we will look further into some of the hyperparameters.
In this workshop, not all of the hyperparameters from the software will be covered.
For those that we don't discuss, we will be using the default settings.
- Max_depth can be an integer or None. It is the maximum depth of the tree. If the max depth is set to None, the tree nodes are fully expanded or until they have less than min_samples_split samples.
- Min_samples_split and min_samples_leaf represent the minimum number of samples required to split a node or to be at a leaf node. Of criteria that need to be met if we want to split new samples in the tree. _Comment: "Of criteria?"_
- Class_weight is important hyperparameter in biology research. If we had a training set and we are using binary classification, i.e., in the T-cells example, if 2 samples are labeled as active and 98 samples are labeled as quiescent, we don't want to train the model that predicts all of the cells to be quiescent. Class_weight parameter would allow putting weight on 2 cells labeled as active. In biology, it is common to have an imbalanced training set more negative than positive instances, so training and evaluating appropriately is essential! The uniform mode gives all classes the weight one. The balanced mode adjusts the weights. _Comment: Split this into a separate paragraph to keep the defintion shorter._

### Step 3 Test and predict

#### Overfitting

If a model fits the training data perfectly, or very well, the model can become too data dependent and not work as well on a new data. 

_Comment: We need more callouts to guide this lesson.  What steps should they take?  What hyperparameter values should they try?  What are the key ideas?  What do they observe about the decision boundaries of a decision tree?_

###  Application in biology

[PgpRules: a decision tree based prediction server for P-glycoprotein substrates and inhibitors](https://doi.org/10.1093/bioinformatics/btz213)
