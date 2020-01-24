---
title: "Decision Trees"
teaching: 10
exercises: 5
questions:
- How does the decision trees classifier make predictions?
objectives:
- Develop knowledge on how a classifier makes predictions. 
- Understand advantages and disadvantages of a classifier. 
- Discuss applying, updating, and evaluating a trained model on new data.
keypoints:
- Decision trees are easy to visualize and intuitive to understand. 
- Decision trees are prone to overfitting.
- In biology, imbalanced datasets are common.
---

### What is the decision tree classifier? 

Decision trees make predictions by asking a sequence of questions for each example and make a prediction based on the responses.
This makes decision trees intuitive.
One of the benefits is that we can clearly see the path of questions and answers we took to get to the final prediction.
For example, a doctor might use a decision tree to decide which medicine to prescribe based on a patient's responses about their symptoms.
Or in the case of T-cells, a decision tree can predict whether a T-cell is active or quiescent.

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decision%20tree1.jpg">
</p>

#### Example

To better understand the algorithm, let's consider the house price prediction example from the [Introduction episode](https://gitter-lab.github.io/ml-bio-workshop/01-introduction/index.html).
We are going to begin with an initial house price range, and for our neighborhood of interest the prices range from $100k - $250k. 
The first question we could ask is the number of bedrooms in the house. 
The answer is 3 bedrooms, and so our new range will be $180k-$250k. 
Then, we will ask about the number of bathrooms, and the answer is 3 bathrooms. 
The new range range is $220-$250.
Finally, we will ask the house's neighborhood.
The answer is Neighborhood A.
That give us the price of $230k. 

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decisiontrees1.jpeg">
</p>

_Comment: Add intructor notes to explain that this is more like a regression example than classification, we use it because it is very intuitive._

#### How does the classifier make predictions?

This intuitive way of understanding decision trees is very close to the way the algorithm is implemented, but we also have the other part of the split to consider. 
Each split of a decision tree classifies instances based on a test of a single feature. 
This test can be True or False. 
The splitting goes from the root at the top of the tree to a leaf node at the bottom.

> ## Definitions
>
> Root node - the topmost node where the decision flow starts.
>
> Leaf node - a bottom node that doesn't split any further. It represents the class label.
{: .callout}

An instance is classified starting from the root and testing the feature specified by the node, then going down the split based on the outcome of the test and testing a different feature specified by another node.  
The graphic shows the full decision tree used for the housing example above.

<p align="center">
<img width="750" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decisiontrees2.jpeg">
</p>

### About the classifier

A decision tree is a supervised learning classifier.
It splits the initial population depending on a certain rule.
The goal of the classifier is to predict the class label on a new set of data based on the rule that the classifier learned from the features of the training examples.
An important property of the decision tree is the __depth of tree__.

> ## Definition
>
> Depth of tree - the number of times we make a split to reach a decision.
{: .callout} 

Some pros of using decision trees:

- easy to visualize
- the classification can be visually followed, so it is easy to reproduce
- makes few assumptions about the data
- can ignore useless features

Some cons of using decision trees:

- prone to __overfitting__ _Comment: Define overfitting_
- requires a way to turn numeric data into a single decision rule

### Step 1 Select data

> ## Software
>
> Let's load *simulated_t_cells_7.csv* data set into the software.
>
> This dataset is engineered specifically to demonstrate the decision tree classifier.
{: .checklist}

> ## Conceptual Questions
>
> What are we trying to predict? 
>
> What is the decision rule?
{: .challenge}

We will continue working on the T-cells example.
The goal is the same, predicting whether a cell is active or quiescent.
We also have the same two features: cell size and intensity. 

### Step 2 Train classifiers

In the original T-cells example, we left the hyperparameters settings as the defaults.
Now we will look further into some of the hyperparameters.
In this workshop, not all of the hyperparameters from the software will be covered.
For the hyperparameters that we don't discuss, use the default settings.
- Max_depth can be an integer or None. It is the maximum depth of the tree. If the max depth is set to None, the tree nodes are fully expanded or until they have less than min_samples_split samples.
- Min_samples_split and min_samples_leaf represent the minimum number of samples required to split a node or to be at a leaf node.
- Class_weight is important hyperparameter in biology research. If we had a training set and we are using binary classification, we don't want to only predict the most abundant class.  For example, in the T-cells example, if 2 samples are active and 98 samples are quiescent, we don't want to train a model that predicts all of the cells to be quiescent. Class_weight parameter would allow putting weight on 2 cells labeled as active so that predicting them incorrectly would be penalized more. 
In biology, it is common to have this type of imbalanced training set with more negative than positive instances, so training and evaluating appropriately is essential! The uniform mode gives all classes the weight one. The balanced mode adjusts the weights.

Without changing any hyperparameter settings, look at the Data Plot.

> ## Think-Pair-Share
>
> What do you notice?
{: .challenge}

_Comment: Should we use a different callout to initially hide these solutions?_
The data plot shows two features, where the blue data points represent the quiescent cells, and the red data points represent the active cells. 

> ## Questions
>
> So if you notice these cut offs, what do you think they represent?
{: .challenge}

They each represent a node in the decision tree. 
When we are trying to come up with the decision rule, we will consider the features and the data plot. 

> ## Conceptual Questions
> 
> What hyperparameter might be important for this example?
{: .challenge}

The given dataset is imbalanced with more quiescent cells than active cells.

> ## Software
>
> Let's change the class_weight to balanced. 
{: .checklist}

> ## Activity
>
> Did this make any difference? 
>
> How does the data plot look for the uniform class_weight and how does it look for the balanced class weight?
{: .challenge}

> ## Play time
>
> Change the max_depth paramenter. 
>
> Did you notice any difference?
{: .challenge}

### Step 3 Test and predict

#### Overfitting

It is easy to go to deep in the tree, and to fit the parameters that are specific for that training set, rather than to generalize to the whole dataset.
This is overfitting. 

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/decisiontree3.jpeg">
</p>

> ## Software
>
> To check if the classifier overfit, first look at the training data. 
>
> Switch between training data and validation data in the upper right corner.
{: .checklist}

By looking at the evaluation metrics and the confusion matrix we can see that when the training data evaluation metrics were perfect, but they were not as great on the validation data.
The classifier probably overfit.

> ## Definitions
>
> Evaluation metrics - used to measure the performance of a model.
{: .callout}

> ## Software
>
> Let's go to the Step 3 in the software. 
{: .checklist}

> ## Questions
>
> Based on accuracy, which classifier was best-performing? 
> 
> Did the classifier overfit?
{: .challenge}

#### Evaluation

Is there anything else that you would want to do before using this to classify the T-cells?

**Are the models reusable?**
We use supervised learning to build our model. We want to be able to use the model on the different data. 

###  Application in biology

[PgpRules: a decision tree based prediction server for P-glycoprotein substrates and inhibitors](https://doi.org/10.1093/bioinformatics/btz213)
