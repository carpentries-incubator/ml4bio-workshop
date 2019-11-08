---
title: "Decision Trees"
teaching: 15
exercises: 0
questions:
- "What is the decision rule?"
objectives:
- "Compare and contrast the strengths and weaknesses of machine learning classifiers commonly used in biology - logistic regression, decision trees, random forests, and neural networks. Assess model selection and recognize that these methods don't necessarily work right out of the box."
- "Discuss applying, updating, and evaluating a trained model on new data."
---

### What is the Decision Tree

Decision trees are part of the CART(Classification and Regression Trees) algorithms. The decision trees are very intuitive and one of the benefits is that we can clearly see the path we took to get to the final answer. We can follow through the whole procedure. This is one of the most popular machine learning algorithms, and it is very intuitive. They are also used across many industries from finances to industrial engineering. For example, a doctor might use a decision tree to decide which medicine to prescribe. Or whether or not a customer qualifies for a credit card. Or in the case of T-cells, whether or not a cell is active or quescent.

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decision%20tree1.jpg">
</p>


### About the Algorithm

Decision tree is a supervised learning classifier. It splits the initial population depending on the a certain rule. The goal of the classifier is the predict the class label on a new set of data based on the rule that the classifier learned from the features. 

**How do we choose a classifier?** (I struggled with this one so please help!)
Pros:
- easy to understand and to visualize

### Step 1 Select Data

Let's load a T-cells #2 data set into our software. This dataset is engineered specifically for the decision tree classifier. Please note, that the prediction metrix for this dataset would rarely reflect the real-world data. 

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the decision tree look graphically?
{: .challenge}

Sample answer for the visual


### Step 2 Train Classifiers

In the original T-cells example we left the hyperparameters settings as default. Now we will look further into some of the parameters. In this workshop not all of the hyperparameters will be covered. 
- Max depth 
- Max samples split/leaf counts of criteria that need to be met if we want to split new samples in the tree talk about
- Class weight is important in biology. It is also used by ¾ classifiers so either introduce it here or come back to it. All it is doing - if there is a binary classification, expensive and inexpensive houses, if training set have 2 expensive and 98 inexpensive we don’t want to train the model that predicts all to be inexpensive. Basically put weight on 2 expensive. In biology we have imbalanced dataset with far more negative with positive instances, so training and evaluating appropriately is important!

###  Examples in biology

[PgpRules: a decision tree based prediction server for P-glycoprotein substrates and inhibitors](https://doi.org/10.1093/bioinformatics/btz213)



