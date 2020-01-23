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
 
A machine learning algorithm gets better at its task when it is shown examples as it tries to define general patterns from the examples.

### Your first machine learning model

Let’s say you want to buy a house.
You follow the real estate market for a few months.
During those months you visit 20 different homes.
Every time you visit a house, you write down all the information you learned about the house. 

> ## Question
>
> What are some of the __features__ that you would consider when buying a house?
{: .challenge}

> ## Definition
>
> Feature - text
{: .callout}

### Group activity

We'll create a housing dataset together.
After the data is collected, we'll label the prices into two categories low or high.

> ## Definitions
>
> Supervised learning - text
>
> Unsupervised learning - text
{: .callout}

Our dataset has labeled data, so our classification is a __supervised machine learning__ task.
_Comment: Classification has not been defined yet.  Add Classification and Regression to definitions above as examples?  Add Clustering as unsupervised example?_
There are different types of machine learning algorithms - supervised learning and __unsupervised learning__ are the main two.
In this workshop, we focus on supervised learning __algorithms__. 

> ## Definitions
>
> Algorithm - text
{: .callout}

Note there can be different types of features (text, number, category, etc).

<p align="center">
<img width="550" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/intro_1.jpg">
</p>

We find a dream home that has not yet been listed for sale and is not part of our existing dataset.
This home is new data, and we want to predict whether the price will be high or low.
Based on the features that we already know, we want to classify our future home in one of the two possible categories. 

We can visualize what the classifier will predict (H for high, L for low price) based on the feature values.

<p align="center">
<img width="550" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/intro_2.jpg">
</p>

This is a simple example of machine learning.

### What is the difference between a machine learning algorithm and a traditional algorithm?

_Comment: Revise this to follow our housing example where we learned the rule from labeled data.  Traditional algorithm is direct implementation based on realtor's expert knowledge._

**Traditional algorithm:**
Let’s say you are doing an experiment, and you need to mix up some solutions.
You have all the ingredients, you have the “recipe” or the proportions, and you follow the recipe to get a solution.

**ML algorithm:**
You are given the ingredients and the final solution, but you don’t know the recipe.
So, what you need to do it to find the “fitting” of the ingredients, that would result in your solution.  


We can think about the following questions whenever we encounter a situation involving machine learning.

> ## Conceptual Questions
>
> What is the benefit of machine learning in this situation?
>
> What are we trying to predict? 
>
> What features do we have for those predictions?
>
> What classifier did we use, and why did we decide to use that one? _Comment: Need to define classifier first._
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

_Comment: What do we do with these papers?  Should ask a question or have partners pick one and discuss the benefit of ML in this situation?_

{% include links.md %}

