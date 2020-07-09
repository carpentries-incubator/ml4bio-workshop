---
title: "Logistic Regression"
teaching: 15
exercises: 15
questions:
- What is linear separability?
objectives:
- Understand advantages and disadvantages of a classifier.
- Develop an ability to discern between linear and nonlinear classifiers.
- Learn various methods to prevent overfitting.
keypoints:
- Logistic regression is a linear classifier.
- The output of logistic regression is probability of a certain class.
- Logistic regression is characterized by linear separability of the data.
- Regularization can be used to improve an overfitting model.
mathjax: true
---

Logistic regression is a classifier that models the probability of a certain label.
In our original example, when we predicted whether a price for a house is high or low, we were classifying our responses into two categories.
Using the logistic regression to predict one of the two labels is a binary logistic regression.
Everything that applies to the binary classification could be applied to multi-class problems (for example, high, medium, or low).
The focus of this workshop is on binary classification.

#### Linear regression vs. logistic regression

You should be familiar with $y=mx+b$. 
If we rewrite it in terms of feature weights $y=w_1x+w_0$ where $w_0$ is the intercept of the line and $w_1$ is the slope of the line. 
In statistics, for the simple linear regression we write intercept term first $y=w_0+w_1x$. 

> ## Definition
>
> Feature weights - determine the importance of a feature in a model.  
{: .callout}

The intercept term is a constant and it is defined as the mean of the outcome when the input is 0. 
This interpretation gets more involved with multiple inputs, but that is out of the scope of the workshop. 
The slope is a feature weight. 
Think about the house price example.
This time we want to predict the house price with one feature, for example, square footage. 
The feature weight is the coefficient for the feature $x$ and it represents the average increase in the house price with a one unit increase in $x$. 
Linear regression predicts the expected house price for the given feature, here the square footage.

When we have multiple features, the linear regression would be $y = w_0 + w_1x1 + w_2x_2 +...+ w_nx_n$.

<p align="center">
<img width="650" src="https://github.com/gitter-lab/ml-bio-workshop/blob/gh-pages/assets/linear_vs_logit.jpg?raw=true">
</p>

### What is logistic regression?

Logistic regression returns the probability that a combination of features with weights belongs to a certain class.

In the visual above that compares linear regression to logistic regression, we can see the "S" shaped function is defined by $p = \frac{1}{1+e^{-(w_0+w_1x_1)}}$. 
The "S" shaped function is the inverse of the logistic function of __odds__. 
It guarantees that the outcome will be between 0 and 1. 
Recall that probability is always between 0 and 1.

> ## Definition
>
> Odds - probability that an event happens divided by probability that an event doesn't happen.
{: .callout}

$odds = \frac{P(event)}{1-P(event)}$

In the original example, if the house price was higher than $150K, it was classified as high, otherwise it was classified as low. 
Consider the situation where we want to predict the likelihood of a certain house price. 

The goal is to predict the probability that the price is high $P(y=high|x;w)$.

From the previous visual, we saw that the sigmoid function represents the probability of a certain outcome. 

$P(y=high|x;w) = \frac{1}{1+e^{-(w_0+w_1x_1)}}$ If we multiply both sides by $1+e^{-(w_0+w_1x_1)}$
$P(y=high|x;w)(1+e^{-(w_0+w_1x_1)}) = 1$
$P(y=high|x;w)+P(y=high|x;w)e^{-(w_0+w_1x_1)} = 1$
$P(y=high|x;w)e^{-(w_0+w_1x_1)} = 1 - P(y=high|x;w)$
$e^{-(w_0+w_1x_1)} = \frac{1-P(y=high|x;w)}{P(y=high|x;w)}$ Take an inverse
$(e^{-(w_0+w_1x_1)})^{-1}=(\frac{1-P(y=high|x;w)}{P(y=high|x;w)})^{-1}$
$\frac{P(y=high|x;w)}{1-P(y=high|x;w)}= e^{w_0+w_1x_1}$ Take the log of both sides.
$log(\frac{P(y=high|x;w)}{1-P(y=high|x;w)})= log (e^{w_0+w_1x_1}) = w_0+w_1x_1$

Remember, $y = w_0 + w_1x1$

Since, we are predicting the probability that the price belongs to one of the two classes, the classification is binary. 

Examine the single feature, that is that square footage of a house, and how it affects the classification whether the house price is high or low. Build the visual of the house price example with one feature.

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/logit1.jpeg">
</p>

<p align="center">
<img width="400" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/logit3.jpeg">
</p>


> ## Conceptual Questions
>
> What rule is the classifier learning?
{: .challenge}

> ## Activity
>
> You are looking to purchase a house. How many square feet do you want?
>
> Is the price for your house high or low based on this one feature?
{: .challenge}


<p align="center">
<img width="650" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/logit2.jpeg">
</p>


Now, let's think about the T-cells example.
If we focus only on one feature, for example cell size, we can use logistic regression to predict the probability that the cell would be active.

The logistic function of odds is a sum of the __weighted features__.
This makes the log-odds function a linear function.
What is important to understand is that the change in one feature by a unit changes the odds ratio.
So logistic regression treats each feature independently.
This affects what type of rules it can learn.

So, an important characteristic of logistic regression features is how they affect the probability.
If the feature weight is positive then the probability of the outcome increases, for example the probability that a T-cell is active increases.
If the feature weight is negative then the probability of the outcome decreases, or in our example, the probability that a T-cell is active decreases.

If we have 2 features, both cell size and cell intensity, logistic regression is learning a different rule.
The rule is a single straight line.
We can control slope and steepness, from class 1 and class 2.

> ## Definition
>
> Linear Separability - drawing a line in the plane that separates all the points of one kind on one side of the line, and all the point of the other kind on the other side of the line.
{: .callout}

Recall, $y=mx+b$ graphed on the coordinate plane. 
It is a straight line. 
$y = w_0 + w_1x_1$ is a straight line. 
The separating function has the equation $w_0+w_1x_1= 0$.
If $w_0+w_1x_1>0$ the house price is classified high, and if $w_0+w_1x_1<0$ the house price is classified as low. 

> ## Think-Pair-Share
>
> Think of an example when a linear separability is not a straight line.
>
> When do you think something is linearly separable?
{: .challenge}

### Step 1 Select Data

> ## Software
>
> Let's load toy_data_3 data set into our software.
>
> This dataset is engineered specifically for the logistic regression classifier.
{: .checklist}

#### Linear separability

> ## Software
>
> Let's train logistic regression classifier.
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

#### Logistic regression vs. Random forests

> ## Software
>
> Let's load _toy_data_8_ data set into our software.
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

> ## Question
>
> How did the accuracy change?
{: .challenge}

This example demonstrates that logistic regression performs great with data that is linearly separable.
However, with the nonlinear data, random forests will be the better choice for a classifier.

### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered.
For those that we don't cover, we will use the default settings.
- Penalty can be L1 or L2 norm.
This parameter is used in regularization. Although there is an option not to use reguralization, in this workshop we will always use reguralization, following common practices in real applications.
- C is the inverse of the regularization strength. Smaller values of C mean stronger regularization.

#### Regularization

> ## Software
>
> Let's load _simulated_t_cells_1_ data set into our software.
>
> This data set is engineered specifically to demonstrate the effects of regularization.
{: .checklist}

Previously, we talked about the positive and negative effect a feature and its weight can have on the outcome probability.
As with decision trees and random forests, logistic regression can overfit.
If we have a complex model with many features, our model might have high variance.

One way to deal with this is regularization.
Regularization can help us decide how many features are too many or too few.
Regularization does not make models fit better on the training data, but it helps with generalizing the pattern to new data.

Recall: C is the inverse of regularization strength.

L1 reguralization or lasso reguralization shrinks the weights of less important features to be exactly 0.
These features are then not used at all to make predictions on new data.

L2 reguralization or ridge reguralization makes the weights of less important features to be very small values.
The higher the C, the smaller the feature weights.

> ## Software
>
> First, set penalty to ‘L1’.
> Experiment with C = 0.08, 0.1, 0.2, 0.5, 1.
>
> Next, set penalty to ‘L2’.
> Experiment with the same set of C.
{: .checklist}

> ## Questions
>
> What do you observe?
>
> Think about the difference.
>
> How does the decision boundary change?
{: .challenge}

L1 will regularize such that one feature weight goes to 0.
We can see the classifier ignores that feature in its decision boundary.

{% include links.md %}
