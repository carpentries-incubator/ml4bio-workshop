---
title: "Decision Trees"
teaching: 15
questions:
- "What is the decision rule?"
objectives:
- "Compare and contrast the strengths and weaknesses of machine learning classifiers commonly used in biology - logistic regression, decision trees, random forests, and neural networks. Assess model selection and recognize that these methods don't necessarily work right out of the box."
- "Discuss applying, updating, and evaluating a trained model on new data."
---

Decision trees are part of the CART(Classification and Regression Trees) algorithms. The decision trees are very intuitive and one of the benefits is that we can clearly see the path we took to get to the final answer. We can follow through the whole procedure. This is one of the most popular machine learning algorithms, and it is very intuitive. They are also used across many industries from finances to industrial engineering. For example, a doctor might use a decision tree to decide which medicine to prescribe. Or whether or not a customer qualifies for a credit card. Or in the case of T-cells, whether or not a cell is active or quescent.

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decision%20tree1.jpg">
</p>

To better understand the algorithm, let's consider a real-life example. 
(add the visual)

### About the Algorithm

Decision tree is a supervised learning classifier. It splits the initial population depending on the a certain rule. The goal of the classifier is the predict the class label on a new set of data based on the rule that the classifier learned from the features. 

### Step 1 Select Data

Let's load a T-cells #2 data set into the software. This dataset is engineered specifically for the decision tree classifier. Please note that the prediction matrix for this dataset would rarely reflect the real-world data. 

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the decision tree look graphically?
{: .challenge}

Sample answer for the visual


### Step 2 Train Classifiers

In the original T-cells example, we left the hyperparameters settings as default. Now we will look further into some of the parameters. In this workshop, not all of the hyperparameters from the software will be covered. Those that we don't cover, we will be using the default settings. 
- Max_depth can be an integer or None. It is the maximum depth of the tree. If the max depth is set to None, the tree nodes are fully expanded or until they have less than min_samples_split samples.
- Min_samples_split and min_samples_leaf represent the minimum number of samples required to split a node or to be at a leaf node. Of criteria that need to be met if we want to split new samples in the tree.
- Class_weight is important hyperparameter in biology research. If we had a training set and we are using binary classification, i.e., in the T-cells example, if 2 samples are labeled as active and 98 samples are labeled as quiescent, we don't want to train the model that predicts all of the cells to be quiescent. Class_weight parameter would allow putting weight on 2 cells labeled as active. In biology, it is common to have an imbalanced training set more negative than positive instances, so training and evaluating appropriately is essential! The uniform mode gives all classes the weight one. The balanced mode adjusts the weights.

### Step 3 Test and Predict

(finish this)

###  Application in biology

[PgpRules: a decision tree based prediction server for P-glycoprotein substrates and inhibitors](https://doi.org/10.1093/bioinformatics/btz213)




