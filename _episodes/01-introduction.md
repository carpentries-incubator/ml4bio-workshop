---
title: "Introduction"
teaching: 15
exercises: 0
questions:
- What is machine learning?
objectives:
- Identify current problems in computational biology and recognize the benefits of machine learning techniques.
---

> ## Think-Pair-Share
> What do you think of when you hear the phrase machine learning?
{: .challenge}


### What is machine learning?

Machine learning is a set of methods that can automatically detect patterns in data and then use those patterns to make predictions on future data or perform other kinds of decision making under uncertainty. 
A machine learning algorithm gets better at its task when it is shown examples as it tries to define general patterns from the examples.


### What does it mean for biology? 

Machine learning is making computation more **sustainable**.
Computers can **scale**, easily making predictions on a large number of items.
It can be very slow and expensive for an expert biologists to manually make the same decisions or manually construct a decision-making model.
Training and executing a machine learning model can be faster and cheaper.

Let’s look at some examples of how machine learning is being used in biology research.
_Comment: let's mostly on biology and remove some of the examples from personal health and medicine.  We'll need to replace some of these._

* Fitness trackers.
* MonBaby tracks your baby’s breathing movements and sleep position, transmitting important alerts directly to your smartphone. Made to help prevent SIDS.
* [Solid-state lithium ion batteries](https://tomkat.stanford.edu/research/designing-better-battery-machine-learning)  
* [Rxrx.ai](https://www.rxrx.ai/) “RxRx1 is a dataset consisting of 296 GB of 16-bit fluorescent microscopy images, the result of the same experimental design being run multiple times with the primary differences between experiments being technical noise unrelated to the underlying biology." 
* [Insilico](https://insilico.com/#rec41711523) extends healthy longevity through innovative AI solutions for drug discovery and aging research. Drug discovery challenges are dominated by machine learning researchers. 
* [Machine learning classifies cancer](https://www.nature.com/articles/d41586-018-02881-7)
* [Finding glaucoma in color fundus photographs using deep learning](https://jamanetwork-com.ezproxy.library.wisc.edu/journals/jamaophthalmology/fullarticle/2749327?resultClick=1)


### Your first model

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

This about these questions and return to them to answer. 

Let’s say you want to buy a house.
You follow the real estate market for a few months.
During those months you visit 20 different homes.
Every time you visit a house, you write down all the information you learned about the house. 

> ## Question
>
> What are some of the features that you would consider when buying a house?
{: .challenge}

Recreate the table and collect the data. _Comment: What does collect mean?  They should write in answers?  Do we do this as a group?_
After the data was collected, label the prices into two categories low or high.
Our dataset has labeled data, so our classification is a supervised machine learning task.
There are different types of machine learning algorithms - supervised learning, unsupervised learning, and reinforcement learning.
In this workshop, we focus on supervised learning algorithms. 

Note different types of features (text, number, category, etc).
_Comment: We should have some example rows filled in already. Shoudl we remove the third ... column to make it 2 features?_
<p align="center">
<img width="550" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/IMG_0016.jpg">
</p>

We find a dream home that is not a part of our already existing data set.
This home is new data, and we want to predict whether the price will be high or low.
Based on the features that we already know, we are trying to classify our future home in one of the two possible categories. 

**Sample answer**
_Comment: label the x and y axis._
<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/78274.jpg">
</p>

> ## Question
>
> What classifier did we use, and why did we decide to use that one?
{: .challenge}

This is a simple example of machine learning.


### Evaluation

Is there anything else that you would want to do before using this to price a house?

**Are the models reusable?**
We use supervised learning to build our model, but the goal is to this be used in transfer learning – that someone else takes our model, and apply it to their problem.
_Comment: Not sure whether this fits the transfer learning definition.  We would likely continue using it in the same region from which data was collected.  Transfer learning is too complex a concept for this workshop._

### What is the difference between a machine learning algorithm and a traditional algorithm?

**Traditional algorithm:**
Let’s say you are doing an experiment, and you need to mix up some solutions.
You have all the ingredients, you have the “recipe” or the proportions, and you follow the recipe to get a solution.

**ML algorithm:**
You are given the ingredients and the final solution, but you don’t know the recipe.
So, what you need to do it to find the “fitting” of the ingredients, that would result in your solution.  

_Comment: What will the key points be?_

{% include links.md %}

