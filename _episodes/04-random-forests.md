---
title: "Random Forests"
teaching: 30
questions:
- ""
objectives:
- Develop knowledge on how a classifier makes the prediction.
- Come up with the decision rule.
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

In this episode, we will learn about the random forests classifier. 
Using the software, we will analyze the pre-engineered dataset and discuss the hyperparameters. 
Then we will consider some applications in biology. 

We previously examined decision trees. 
One of their main weaknesses is their tendancy to overfit if the tree is allowed to get too deep. 
In training a decision tree we often have to therefore alter the decison tree's parameters to keep it a "generalist", instead of allowing it to overly specialize and overfit. *Milica's Comment: I struggled with this... can we find a better way to say this?*

Random forests deals with the problem of overfitting by creating multiple trees, with each tree trained slighly differently so it overfits differently.
Random forests is a classifier that combines a large number of decision trees.
The decisions of each tree are then combined to make the final classification.
This "team of specialists" approach random forests take often outperforms the "single generalist" approach of decision trees. *Milica's Comment: Same as the above, can we reword this somehow?*

### Why is it called "random" forests?

If we, when training each tree in the forest, give every tree the same data, we would get the same predictions that are prone to overfitting. 
In order to train the decision trees differently we need to provide slightly different data to each tree. 
To do this, we choose a **random** subset of the data to give to each tree. 
When training at each node in the tree we also **randomize** which features can be used to split the data.
The final prediction is based on a vote or the average taken across all the decision trees in the forest.


### Step 1 Select data

> ## Software
>
> Load a T-cells #3 data set into our software. 
> This dataset is designed to specifically illustrate properties of the random forest classifier.
{: .checklist}

> ## Conceptual Questions
>
> What are we trying to predict? 
>
> What is the decision rule?
>
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
