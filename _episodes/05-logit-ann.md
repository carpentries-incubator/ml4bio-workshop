---
title: "Logistic Regression, Artificial Neural Networks, and Linear Separability"
duration: 45
questions:
- What is linear separability?
objectives:
- Understand advantages and disadvantages of logistic regression and artificial neural networks
- Distinguish linear and nonlinear classifiers
- Predict whether a dataset is likely to be linear or nonlinear
keypoints:
- Logistic regression is a linear classifier.
- The output of logistic regression is probability of a certain class.
- Artificial neural networks can be viewed as an extension of logistic regression
- Artificial neural networks can have nonlinear decision boundaries
mathjax: true
---

## Logistic Regression

Logistic regression is a classifier that models the probability of a certain label.
In the T-cells example, we were classifying whether cells were in the two categories of active or quiescent.
Using the logistic regression to predict one of the two labels is a binary logistic regression.
Everything that applies to the binary classification could be applied to multi-class problems (for example if there was a third cell state).
We will be focusing on the binary classification problem.

#### Linear regression vs. logistic regression

We can write the equation for a line as $y=mx+b$, where $x$ is the x-coordinate and $y$ is the y-coordinate.
If we rewrite it in terms of feature weights $y=w_1x+w_0$ where $w_0$ is the intercept of the line and $w_1$ is the slope of the line. 
In statistics, for the simple linear regression we write intercept term first $y=w_0+w_1x$. 

> ## Definition
>
> Feature weights - determine the importance of a feature in a model.  
{: .callout}

The intercept term is a constant and it is defined as the mean of the outcome when the input is 0. 
This interpretation gets more involved with multiple inputs, but that is out of the scope of the workshop. 
The slope is a feature weight. 
In the T-cells example, the feature weight is the coefficient for the feature $x$ and it represents the average increase in the confidence the cell is active with a one unit increase in $x$. 

When we have multiple features, the linear regression would be $y = w_0 + w_1x_1 + w_2x_2 +...+ w_nx_n$.

<p align="center">
<img width="550" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/Linreg-vs-logit_.png">
</p>

### What is logistic regression?

Logistic regression returns the probability that a combination of features with weights belongs to a certain class.

In the visual above that compares linear regression to logistic regression, we can see the "S" shaped function is defined by $p = \frac{1}{1+e^{-(w_0+w_1x_1)}}$. 
The "S" shaped function is the inverse of the logistic function of __odds__. 
It guarantees that the outcome will be between 0 and 1. 
This allows us to treat the outcome as a probability, which also must be between 0 and 1. 

> ## Definition
>
> Odds - probability that an event happens divided by probability that an event doesn't happen.
{: .callout}

$odds = \frac{P(event)}{1-P(event)}$

Now, let's think about the T-cells example.
If we focus only on one feature, for example cell size, we can use logistic regression to predict the probability that the cell would be active.

The logistic function of odds is a sum of the __weighted features__.
Each feature is simply added together with a weight inside the logistic function. 
So logistic regression treats each feature independently.
This means that, unlike decision trees, logistic regression is unable to find interactions between features.
For instance, copy number increase of a certain gene may only affect cell state if another gene is also mutated.
This feature independence affects what type of rules logistic regression can learn.

> ## Questions to consider
>
> What could be another example of 2 features that interact with each other?
{: .challenge}

An important characteristic of features in logistic regression is how they affect the probability.
If the feature weight is positive then the probability of the outcome increases, for example the probability that a T-cell is active increases.
If the feature weight is negative then the probability of the outcome decreases, or in our example, the probability that a T-cell is active decreases.

> ## Definition
>
> Linear Separability - drawing a line in the plane that separates all the points of one kind on one side of the line, and all the points of the other kind on the other side of the line.
{: .callout}

Recall, $y=mx+b$ graphed on the coordinate plane. 
It is a straight line. 
$y = w_0 + w_1x_1$ is a straight line. 
The separating function has the equation $w_0+w_1x_1= 0$.
If $w_0+w_1x_1>0$ the T-cell is classified as active, and if $w_0+w_1x_1<0$ the T-cell is classified as quiescent. 

> ## Questions to consider 
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

> ## Questions to consider - Poll
>
> Look at the Data Plot.
> What do you notice?
>
> Is the data linearly separable?  
{: .challenge}

Logistic regression can also be visualized as a network of features feeding into a single logistic function.

<p align="center">
<img width="750" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/logit_nodes.png">
</p>

At the logistic function, the connected features are combined and fed into the logistic function to get the classifier's prediction.
In this visualization, we can see that the features never interact with each other until they reach the logistic function, which results in feature independence.

## Artificial Neural Networks

An artificial neural network can be viewed as an extension of the logistic regression model, where additional layers of feature interactions are added.
These additional layers allow for more complex, non-linear decision boundaries to be learned.

<p align="center">
<img width="750" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/ann_nodes.png">
</p>

Artificial neural networks have inputs and outputs, just like logistic regression, but have one or more additional layers called __hidden layers__ comprised of __hidden units__. 
Hidden layers can contain any number of hidden units.
In the neural network diagram, each unit in the input, output, and hidden layers resemble neurons in a human brain, hence the name neural network.
The above figure shows a fully connected hidden layer, where every feature interacts with every other feature to form a new value. 
The new value is created by passing the sum of the weighted feature values into the __activation function__ of the hidden layer.
These new values are then fed to the logistic function as opposed to the raw features. 

> ## Definitions
>
> Hidden unit - a function in a neural network which takes in values, applies some function to them, and outputs new values to be used in subsequent layer of the neural network. 
>
> Hidden layer - a layer of hidden units in a neural network. 
>
> Activation function - the function used to combine values in a specific layer of a neural network. 
{: .callout}

#### Linear vs. Nonlinear classifiers

> ## Software
>
> Let's load _toy_data_8_ data set into our software.
>
> This data set is engineered specifically to demonstrate the difference between linear and nonlinear classifiers.
>
> Train a logistic regression classifier using the default hyperparameters.
{: .checklist}

> ## Questions to consider
>
> What are the evaluation metrics telling us?
>
> Look at the Data Plot.
> What do you notice?
>
> Is the data linearly separable?  
{: .challenge}

> ## Software
>
> Now train a neural network on the same data.
{: .checklist}

> ## Questions to consider
>
> What shape is the decision boundary?
>
> How did the performance of the model change on the validation data?
{: .challenge}

This example demonstrates that logistic regression performs great with data that is linearly separable.
However, with the nonlinear data, more complex models such as artificial neural networks need to be used.

### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be discussed.
The most important hyperparameters are:
- The regularization penalty can be L1 or L2.
Although it is possible to train a logistic regression classifier without regularization, regularization is always used in the ml4bio software.
This follows best practices in real applications.
- C is the inverse of the regularization strength.
Smaller values of C mean stronger regularization.

For those hyperparameters that we don't cover, we will use the default settings.

#### Artificial neural networks in practice

Artificial neural networks with a single hidden layer tend to perform well on simpler data, such as the toy datasets in this workshop. 

However artificial neural networks used on more complex data, such as raw image data or protein structure data, typically have much more complex architectures. 

For example, below is the architecture of a neural network used to [evaluate the skill of robotic surgery arms](https://link.springer.com/article/10.1007/s11548-018-1860-1) based on motion captured over time.

<p align="center">
<img width="750" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/fig/third_party_figures/example_deep_network.png">
</p>

This neural network still consists of hidden layers combined with functions, but contains many specialized layers which perform specific operations on the input features.

> ## Classifier selection scenarios - Poll
>
> In the following scenarios, which classifier would you choose?
>
> 1. You are interested in learning about how different factors contribute to water preservation adaptations in plants. 
> You plan to create a model for each of 4 moisture preservation adaptations, where in each model the presence of the moisture preservation adaptation is the class being predicted. 
> You use a dataset of 200 plant species to train each model.
> You have 15 features for each species, consisting of environmental information such as latitude, average temperature, average rainfall, average sun intensity, etc. 
>
> 2. You have been tasked with creating a model to predict whether a mole sample is benign or malignant based on gene expression data. 
> Your dataset is a set of 380 skin samples, each of which has expression data for 50 genes believed to be involved in melanoma. 
> It is likely that a combination of genes is required for a mole to be cancerous. 
>
{: .challenge}

#### Regularization

Recall, regularization is used to make sure that our model pays attention only to the important features to avoid overfitting. 
Previously, we talked about the positive and negative effect a feature and its weight can have on the outcome probability.
As with decision trees and random forests, logistic regression can overfit.
If we have a complex model with many features, our model might have high variance.
Regularization can help us decide how many features are too many or too few.
Regularization does not make models fit better on the training data, but it helps with generalizing the pattern to new data.

> ## Definition
>
> Regularization - approaches to make machine learning models less complex in order to reduce overfitting.
>
> Penalty - a regularization strategy that reduces the importance of certain features by adding a cost to the feature weights, which makes the feature weights smaller.
{: .callout}

#### L1 regularization

L1 regularization is also known as Lasso reguralization.

> ## Software
>
> Let's load _simulated_t_cells_1_ data set into our software.
>
> This data set is engineered specifically to demonstrate the effects of regularization.
{: .checklist}

Recall that $x_1, x_2, ..., x_n$ are the features, and $w_0, w_1, ..., w_n$ are the feature weights.
Without regularization, the classifier might fit the training data perfectly, giving certain values to each weight that would lead to overfitting. 
This model could be very complex and it would generalize poorly on the future data.

L1 regularization prevents overfitting by adding a penalty term and mathematically shrinking (decreasing) some weights.
L1 could shrink the weights of less important features all the way to 0, effectively deleting those weights. 
The corresponding features are then not used at all to make predictions on new data.

Recall: C is the inverse of regularization strength.
Smaller values of C will shrink more weights and use fewer features to make the prediction.

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

#### L2 Penalty

L2 regularization is also known as ridge reguralization.
L2 reguralization makes the weights of less important features to be small values.
Unlike L1 regularization, L2 regularization does not necessarily shrink the weights to 0.
The higher the value of C, the smaller the feature weights will be.

> ## Break
> Let's take a short break. 
>
{: .discussion}

{% include links.md %}
