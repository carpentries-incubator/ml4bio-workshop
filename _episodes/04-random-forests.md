---
title: "Random Forests"
teaching: 15
exercises: 15
questions:
- How is the classifier preventing overfitting?
objectives:
- Develop knowledge on how a random forest classifier makes a prediction.
- Come up with the random forest decision rule.
keypoints:
- Random forests combines the large number of decision trees.
- Each tree is given a random subset of the data.
- Random forests tend to be a good classifier on many different datasets.
---

In this episode, we will learn about the random forests classifier. 
Using the software, we will analyze the pre-engineered dataset and discuss the hyperparameters. 
Then we will consider some applications in biology. 

We previously examined decision trees. 
One of their main weaknesses is their tendancy to overfit if the tree is allowed to get too deep. 
In training a decision tree we often have to alter the decison tree's parameters to keep it a "generalist", instead of allowing it to overly specialize and overfit the training examples. 

Random forests deals with the problem of overfitting by creating multiple trees, with each tree trained slighly differently so it overfits differently.
Random forests is a classifier that combines a large number of decision trees.
The decisions of each tree are then combined to make the final classification.
This "team of specialists" approach random forests take often outperforms the "single generalist" approach of decision trees. 
Multiple overfitting classifiers are put together to reduce the overfitting.

### Motivation from the bias variance trade-off

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decisiontree3.jpeg">
</p>

In the previous lesson we looked at overfitting. 
Looking again at the different decision boundaries, note that the one of the left has high __bias__ while the one on the right has high __variance__. 

> ## Definitions
>
> Bias - The assumptions made by a model about what the decision boundary will look like. Models with high bias are less sensitive to changes in the training data.
>
> Variance - The amount the training data affects what a model's decision boundary looks like. Models with high variance have low bias. 
>
> Note that these concepts have more exact mathmatical definitions which are beyond the scope of this workshop.
{: .callout}

Random forests are based on mitigating the negative effects of this trade-off by creating multiple high variance models that work together.


### Why is it called "random" forests?

If when training each tree in the forest, we give every tree the same data, we would get the same predictions that are prone to overfitting. 
In order to train the decision trees differently we need to provide slightly different data to each tree. 
To do this, we choose a **random** subset of the data to give to each tree. 
When training at each node in the tree we also **randomize** which features can be used to split the data.
The final prediction is based on a vote or the average taken across all the decision trees in the forest.


### Step 1 Select data

> ## Software
>
> Load a simulated_t_cells_2 data set into our software. 
{: .checklist}

> ## Conceptual Questions
>
> What are we trying to predict? 
>
> What is the decision rule?
{: .challenge}


### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered.
For the hyperparameters that we don't discuss, use the default settings. 
- N-estimator represents the number of the decision trees that go into forest. Although we want to consider the biggest number of trees possible, there is a certain number where the classifier performance won't be improving. The software allows at most 50 trees.
- Max-features represents the number of features considered. For the classification problems *sqrt* is mostly used. That is the square root of the total number of features.

> ## Software
>
> Train the classifier without changing the hyperparameters. 
{: .checklist}

> ## Think-Pair-Share
>
> What happens when you change the number of trees in the forest?
>
> Compare test data and validation data.
{: .challenge}


### Step 3 Test and Predict

> ## Software
>
> In the software, go to the Step 3. 
>
> Choose the evluation metric to select the best-performing classifier. 
{: .checklist}


{% include links.md %}
