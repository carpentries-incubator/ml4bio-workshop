---
title: "Introduction"
duration: 35
questions:
- What is machine learning?
objectives:
- Define and give examples of machine learning
- Identify problems in computational biology suitable for machine learning
keypoints:
- Machine learning algorithms recognize patterns from example data
- Supervised learning involves predicting labels from features
---

### What is this workshop about?

The learning objectives of this workshop are:

1. Identity and characterize machine learning and a machine learning workflow.

2. Evaluate whether a particular problem is easy or hard for machine learning to solve. 

3. Assess a typical machine learning methodology presented in an academic paper. 

4. Gain confidence in and appreciation for machine learning in biology.

We will also be learning about some specific machine learning models: decision trees, random forests, logistic regression, and artificial neural networks. 



### What is machine learning?

> ## Definition
>
>__Machine learning__ is a set of methods that can automatically detect patterns in data and then use those patterns to make predictions on future data or perform other kinds of decision making under uncertainty.
{: .callout}

A machine learning __algorithm__ gets better at its task when it is shown examples as it tries to define general patterns from the examples.

One of the most popular textbooks on machine learning, *Machine Learning* by Tom Mitchell, defines machine learning as, "the study of computer algorithms that improve automatically through experience."

> ## Definition
>
> Algorithm - is a relationship between input and output. It is a set of steps that takes an input and produces an output.
{: .callout}

Machine learning can be broadly split into two categories, __supervised machine learning__ and and __unsupervised learning__.
Machine learning is considered supervised when there is a specific answer the model is trying to predict, questions like what is the price of a house? 
Is this mushroom edible? 
What disease does this patient have. 
Some examples of supervised machine learning are __classification__ and __regression__.
These will be further defined in the next lesson.

Unsupervised machine learning has no particular target for it's learning, it is instead trying to answer general questions about patterns.
This can include questions like how do these cells group together?
In what way are these samples most different from each other?
An example of unsupervised machine learning is __clustering__.
Unsupervised learning will not be covered in this workshop.
For some external resources, check out the [glossary][lesson-glossary].

> ## Definitions
>
> Supervised learning - training a model from the labeled input data.
>
> Unsupervised learning - training a model from the unlabeled input data to find patterns in the data. 
{: .callout}

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/supervised_vs_unsupervised.png">
</p>

> ## Definition
>
> Clustering - grouping related samples in the data set. In the house price example, a sample is each house.
>
> Classification - predicting a category for the samples in the data set. In the house price example, the categories are high and low.
> 
> Classifier - a specific model or algorithm that performs classification.
>
> Regression - predicting a continuous number for the samples in the data set. 
>
> Model - mathematical representation that generates predictions based on the input data.
{: .callout}

> ## Scenarios
> 1. You are trying to understand how temperature affects the speed of embryo development in mice.
> After running an experiment where you record developmental milestones in mice at various temperatures, you run a linear regression on the results to see what the overall trend is. 
> You use the regression results to predict how long certain developmental milestones will take at temperatures you’ve not tested. 
>
> 2. You want to create a guide for which statistical test should be used in biological experiments. 
> You hand-write a decision tree based on your own knowledge of statistical tests. 
> You create an electronic version of the decision tree which takes in features of an experiment and outputs a recommended statistical test. 
>
> 3. You are annoyed when your phone rings out loud, and decide to try to teach it to only use silent mode. 
> Whenever it rings, you throw the phone at the floor. 
> Eventually, it stops ringing. “It has learned. This is machine learning,”  you think to yourself. 
{: .callout}


### What is the difference between a machine learning algorithm and a traditional algorithm?

**Traditional algorithm:**
Let’s say you are doing an experiment, and you need to mix up some solutions.
You have all the ingredients, you have the “recipe” or the proportions, and you follow the recipe to get a solution.

**ML algorithm:**
You are given the ingredients and the final solution, but you don’t know the recipe.
So, what you need to do it to find the “fitting” of the ingredients, that would result in your solution.  

Think about the following questions whenever we encounter a situation involving machine learning.

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
* [Predicting 3d protein structures](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
* [Discovering new antibiotics](https://doi.org/10.1016/j.cell.2020.01.021)
* [Recognizing clinical impact of genetic variants](https://doi.org/10.1038/s41588-018-0167-z)

{% include links.md %}
