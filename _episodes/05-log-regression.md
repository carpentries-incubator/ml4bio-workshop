---
title: "Logistic Regression"
teaching: 30
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

Logistic regression is a classifier that models the probability of a certain label. 
In our original example, when we predicted whether a price for a house is high or low, we were classifying our responses into two categories.
If we use logistic regression to predict one of the two labels, it is a binary logistic regression.
Everything that applies to the binary classification could be applied to multi-class problems. 
We will be focusing on binary classification in this workshop.

### What is logistic regression?

Logistic regression returns the probability that a combination of features with weights belong to a certain class. 
NOTE: Probability is always between 0 and 1.
Let's build the visual of the house price example with one feature. 
We want to predict whether the house price is high or low.
We will be predicting the probability that the price belongs to one of the two classes, so this is binary classification. 
Let's look at the single feature, square footage of a house, and how it affects whether the house price is high or low. 
The price is high if it is more than $150k and low if it is less than $150k.

> ## Conceptual Questions
>
> What rule is the classifier learning?
{: .challenge}

*Milica's comment: add visual*

Now, let's think about the T-cells example. 
If we focus only on one feature, for example cell size, we can use logistic regression to predict the probability that the cell would be active. 

*Milica's comment: should we add multiple visuals with different weights?*

If we have 2 features, both cell size and cell intensity, logistic regression is learning a different rule. 
The rule is a single straight line. 
We can control slope, steepness, from class 1 and class 2 

> ## Conceptual Questions
>
> Why is it a straight line?
> What exactly is the classifier learning?
{: .challenge} 
  

### Step 1 Select Data

Let's load a T-cells #4 data set into our software.
This dataset is engineered specifically for the logistics regression classifier.
Please note, that the prediction metrix for this dataset would rarely reflect the real-world data. 

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
{: .challenge}


### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered. Those that we don't cover, we will be using the default settings. 
- Penalty 
- C


### Step 3 Test and Predict

**Finish once the dataset is done**

