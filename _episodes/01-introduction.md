---
title: "Introduction"
teaching: 30
exercises: 0
questions:
- What is machine learning?
objectives:
- Identify current problems in computational biology and recognize the benefits of machine learning techniques.
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
>Think-Pair-Share
>What do you think of when you hear the phrase machine learning?
{: .challenge}


### What is machine learning?

“Machine learning is a set of methods that can automatically detect patterns in data and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty” Murphy Machine Learning

* Machine learning is a machine that gets better at the task when you show it examples. 
* We try to accomplish this by designing tools that learn to solve problems from data. 


### What does it mean for biology? 
It is making computation more sustainable. 
Let’s look at some “mainstream” examples of how machine learning is being used in biology research.
Gene editing and CRISPR (if we use this, be careful!!! And know HOW it is being used)
Fitness trackers – fitbit, polar 
MonBaby - tracks your baby’s breathing movements and sleep position, transmitting important alerts directly to your smartphone. Preventing SIDS
Solid-state lithium ion batteries  https://tomkat.stanford.edu/research/designing-better-battery-machine-learning
There are those who try to prevent aging or at least slow it down like Spring Discovery – “We've built a machine learning platform to accelerate experimentation for discovering therapies for aging and its related diseases.” https://www.springdisc.com/#approach (BE CAREFUL)


### What about research?

Genomics? Number of genes too big? Something with that? 
Biology AI companies
Rxrx.ai “RxRx1 is a dataset consisting of 296 GB of 16-bit fluorescent microscopy images, the result of the same experimental design being run multiple times with the primary differences between experiments being technical noise unrelated to the underlying biology.
“https://www.rxrx.ai/
by Recursion pharmaceuticals “Drug discovery, reimagined through AI”
https://recursionpharma.com
Insilico - extend healthy longevity through innovative AI solutions for drug discovery and aging research.
https://insilico.com/#rec41711523


### Here at Wisconsin:

Center for Predictive Computational Phenotyping


### Your first model

Let’s say you want to buy a house. For majority of us, the price is the main driver in our decision. The problem - after we see the house, how much money should we offer? If we translate that into machine learning language – we want to PREDICT, the price of a house.

>What are some of the features that you would going to consider when buying a house?
{: .challenge}

Note different types of features (text, number, category, etc)

price = features * x

**How do we call this model? Is there anything missing?**

This is the simplest example of machine learning.


### Evaluation part – 

Is there anything else that you would want to do before using this to price a house?


**Are the models reusable?**
We use supervised learning to build our model, but the goal is to this be used in transfer learning – that someone else takes our model, and apply it to their own problem.

**Machine learning algorithm vs. “normal” algorithm**
Normal algorithm:
Let’s say you are doing an experiment, and you need to mix up some solution. You have all the ingredients for the solution, you have the “recipe” or the proportions needed for the solution, so you follow the recipe, to get the solution. 
However, in ML algorithm – you are given the ingredients and the final solution, but you don’t know the recipe. So, what you need to do it to find the “fitting” of the ingredients, that would result in your solution. 


{% include links.md %}

