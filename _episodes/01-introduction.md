---
title: "Introduction"
teaching: 10
exercises: 5
questions:
- What is machine learning?
objectives:
- Define machine learning
- Identify problems in computational biology suitable for machine learning
- Recognize the benefits of machine learning techniques
keypoints:
- Machine learning algorithms recognize patterns from example data
- Supervised learning involves predicting labels from features
---

> ## Think-Pair-Share
> What do you think of when you hear the phrase machine learning?
{: .challenge}

### What is machine learning?

> ## Definition
>
>__Machine learning__ is a set of methods that can automatically detect patterns in data and then use those patterns to make predictions on future data or perform other kinds of decision making under uncertainty.
{: .callout}

A machine learning __algorithm__ gets better at its task when it is shown examples as it tries to define general patterns from the examples.

> ## Definition
>
> Algorithm - is a relationship between input and output. It is a set of steps that takes an input and produces an output.
{: .callout}

### Your first machine learning model

Let’s say you want to buy a house.
You follow the real estate market for a few months.
During those months you visit 20 different houses.
Every time you visit a house, you write down all the information you learned about the house.

> ## Question
>
> What are some of the __features__ that you would consider when buying a house?
{: .challenge}

> ## Definitions
>
> Feature - a property or a characteristic of the observed object. Used as an input.
>
> Class label - prediction output.
> Also called the target variable or label.
{: .callout}

Note there can be different types of features (text, number, category, etc).

### Group activity

Create a housing dataset as a group.
After the data is collected, label the prices into two categories low or high.

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/intro_1.jpg">
</p>

Our dataset has labeled data, so our problem is a __supervised machine learning__ task.
There are different types of machine learning algorithms - supervised learning and __unsupervised learning__ are the main two.
Some examples of supervised machine learning are __classification__ and __regression__.
These will be further defined in the next lesson.
An example of unsupervised machine learning is __clustering__.

> ## Definitions
>
> Supervised learning - training a model from the labeled input data.
>
> Unsupervised learning - training a model from the unlabeled input data to find the patterns in the data.
{: .callout}

> ## Definition
>
> Clustering - grouping related samples in the data set. In the house price example, a sample is each house.
>
> Classification - classifying related samples in the data set into a category. The goal of classification is to predict which category each sample belongs to. In the house price example, the categories are high and low.
> 
> Classifier - a specific model or algorithm that performs classification.
>
> Regression - unlike classification that predicts a category, regression predicts a numeric value, such as the price of a house in dollars.
{: .callout}

In this workshop, we focus on supervised learning algorithms.

We find a dream home that has not yet been listed for sale and is not part of our existing dataset.
This home is new data, and we want to predict whether the price will be high or low.
Based on the features that we already know, we want to classify our future home in one of the two possible categories.

Visualize what the __machine learning model__ will predict (H for high, L for low price) based on the feature values.

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/Intro-dataset.png">
</p>

> ## Definition
>
>__Decision boundary__ - a region where all the patterns within the decision boundary belong to the same class. It divides the space that is represented by all the data points. Identifying the decision boundary can be used to classify the data points. The decision boundary does not have to be linear.
{: .callout}

This is a simple example of machine learning.

> ## Definition
>
> Model - mathematical representation that generates predictions based on the input data.
{: .callout}

### What is the difference between a machine learning algorithm and a traditional algorithm?

**Traditional algorithm:**
Let’s say you are doing an experiment, and you need to mix up some solutions.
You have all the ingredients, you have the “recipe” or the proportions, and you follow the recipe to get a solution.

**ML algorithm:**
You are given the ingredients and the final solution, but you don’t know the recipe.
So, what you need to do it to find the “fitting” of the ingredients, that would result in your solution.  

Think about the following questions whenever we encounter a situation involving machine learning.

> ## Conceptual Questions
>
> What is the benefit of machine learning in this situation?
>
> What are we trying to predict?
>
> What features do we have for those predictions?
>
> What machine learning model did we use, and why did we decide to use that one?
{: .challenge}

### What does machine learning mean for biology?

Machine learning can **scale**, easily making predictions on a large number of items.
It can be very slow and expensive for expert biologists to manually make the same decisions or manually construct a decision-making model.
Training and executing a machine learning model can be faster and cheaper.
Machine learning may also recognize complex patterns that are not obvious to experts.

Let’s look at some examples of how machine learning is being used in biology research.

* [Imputing missing SNP data.](https://doi.org/10.1038/sj.ejhg.5201988)
* [Identifying transcription-factor binding sites from DNA sequence data alone, and predicting gene function from sequence and expression data.](http://doi.org/10.1038/nrg3920)   
* [Finding drug targets in  breast, pancreatic and ovarian cancer.](https://doi.org/10.1186/s13073-014-0057-7)
* [Diagnosing cancer from DNA methylation data.](http://doi.org/10.1038/d41586-018-02881-7)
* [Finding glaucoma in color fundus photographs using deep learning.](http://doi.org/10.1001/jamaophthalmol.2019.3512)
* [Predicting lymphocyte fate from cell imaging](https://doi.org/10.1371/journal.pone.0083251)

{% include links.md %}
