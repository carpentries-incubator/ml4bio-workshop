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


> ## Conceptual Questions
>
> What rule is the classifier learning?
{: .challenge}

add visual 

> ## Activity
>
> You are looking to purchase a house. How many square feet do you want? 
>
> Is the price for your house high or low based on this one feature?
{: .challenge}

Now, let's think about the T-cells example. 
If we focus only on one feature, for example cell size, we can use logistic regression to predict the probability that the cell would be active. 

What the "S" shaped function is represented by the logistic function of __odds__. 

> ## Definitions
>
> Odds - probability that an event happens divided by probability that an event doesn't happen.
{: .callout}

The logistic function of odds is a sum of the weighted features.
This makes the log-odds function a linear function, and logistic regression a linear classifier. 
We will not be going through the math, so trust us on this! 
What is important to understand is that the change in one feature by a unit changes the odds ratio. 
So logistic regression treats each feature independently. 
This affects what type of rules it can learn. 

If we have 2 features, both cell size and cell intensity, logistic regression is learning a different rule. 
The rule is a single straight line. 
We can control slope and steepness, from class 1 and class 2.

add visual 

> ## Definitions
>
> Linear Separability - drawing a line in the plane that separates all the points of one kind on one side of the line, and all the point of the other kind on the other side of the line. 
{: .callout}

> ## Conceptual Questions
>
> Why is it a straight line?
>
> What exactly is the classifier learning?
{: .challenge} 
 

### Step 1 Select Data

> ## Software
>
> Let's load toy_data_3 data set into our software.
>
> This dataset is engineered specifically for the logistics regression classifier.
{: .checklist}

> ## Conceptual Questions
>
> What are we trying to predict? 
{: .challenge}

#### Linear separability

> ## Software
>
> Let's train logistic regresion classifier. 
>
> For now use the default hyperparameters. 
{: .checklist}


> ## Conceptual Questions
>
> Look at the Data Plot. 
> What do you notice? 
>
> Is the data linearly separable?  
{: .challenge} 

#### Logistic regression vs. Ranfom forests

> ## Software
>
> Let's load toy_data_8 data set into our software.
>
> This data set is engineered specifically to demonstrate the difference between linear and nonlinear classifiers.
>
> Train the data set with the default hyperparameters using logistic regression classifier.
{: .checklist}

> ## Questions
>
> What is the evaluation metrics telling?
>
> Look at the Data Plot. 
> What do you notice? 
>
> Is the data linearly separable?  
{: .challenge} 

> ## Software
>
> Train the data set with the default hyperparameters using random forests classifier.
{: .checklist}

> ## Questions
>
> How did the accuracy change? 
{: .challenge} 

This example demonstrates that logistic regression performs great with the data that is linearly separable. 
However, with the nonlinear data, random forests will be the better choice for a classifier.

### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered. Those that we don't cover, we will be using the default settings. 
- Penalty can be l1 or l2 norm. 
This parameter is used in reguralization and, although, there is an option not to use reguralization, in this workshop we will only be focusing on the example when we use reguralization.
- C is an inverse of reguralization strength. Smaller values mean stronger regularization.

#### Regularization

L2 reguralization or ridge reguralization - text

L1 reguralization or lasso reguralization - text


### Step 3 Test and Predict

**Finish once the dataset is done**

