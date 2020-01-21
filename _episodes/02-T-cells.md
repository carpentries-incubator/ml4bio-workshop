---
title: "Classifying T-cells"
teaching: 25
exercises: 5
questions:
- What are the steps in a machine learning workflow?
objectives:
- Define classifiers
- Comprehend how the datasets are preprocessed
- Learn what are samples, features, and class labels in a biological example
- Understand training, validation, and test sets
keypoints:
- Choosing how to represent a dataset in a machine learning task can have a large impact on performance
- The machine learning workflow is split into data preprocessing and selection, training and model selection, and evaluation stages
- Splitting a dataset into training, validation, and testing sets is key to being able to properly evaluate a machine learning method
---

### Why classify T-cells

Let's consider how to answer a real biological question using the same concepts we saw in the introduction.
The question pertains to immunotherapy, a type of cancer treatment that uses the body's immune cells to boost natural defenses against cancer.
T-cells are a common target for immunotherapies.
For immunotherapy to be effective, the modified T-cells must be in an active state.
Here we will study how to assess the activation state of individual T-cells.

Scientists at UW-Madison and the Morgridge Institute [developed an imaging method](https://doi.org/10.1101/536813) to easily and quickly acquire images of T-cell without destroying them.
These images contain information that can be used to predict T-cell activity.
We would like to develop a classifier that can take an image of a T-cell and predict whether it is active or quiescent.
The active cells would then be used for immunotherapy, and the quiescent cells can be considered inactive and would be discarded.
_Comment: Make sure we include Jay in acknowledgements when we update them_

> ## Definitions
>
> Active cells - text
>
> Quiescent cells - text
{: .callout}

### Dataset description

This microscopy dataset includes grayscale images of two type of T-cells: activated and quiescent.
These T-cells come from blood samples from six human donors.

|Activated|Quiescent|
|:---:|:---:|
|![Activated T-cell 1]({{ page.root }}/fig/activated-tcell-1.png)|![Quiescent T-cell 1]({{ page.root }}/fig/quiescent-tcell-1.png)|
|![Activated T-cell 2]({{ page.root }}/fig/activated-tcell-2.png)|![Quiescent T-cell 3]({{ page.root }}/fig/quiescent-tcell-2.png)|
|![Activated T-cell 3]({{ page.root }}/fig/activated-tcell-3.png)|![Quiescent T-cell 3]({{ page.root }}/fig/quiescent-tcell-3.png)|

We will use a subset of the images and follow the workflow in a [T-cell classification study](https://doi.org/10.1002/jbio.201960050).

_Comment: Restructure to be like slides, add 2-3 examples, ask questions about what we think the labels are (show of hands)_
_Comment: If we want to make this an activity, show the first two examples of each class and reserve the third for the interactive question_

### Machine learning methods

The goal of this study is to develop a method to classify T-cell activation state (activated vs. quiescent). 

### ml4bio software setup

We will be using ml4bio software to build classifiers with the T-cell images.
Refer to the [Setup](https://gitter-lab.github.io/ml-bio-workshop/setup.html) for instructions on how to install and launch the ml4bio software. _Comment: We should learn how to use relative links_ 
To better understand the software features, check out the [About ml4bio](https://gitter-lab.github.io/ml-bio-workshop/about/index.html) page.
All of the datasets that we will be using for this workshop have been formatted to fit the ml4bio requirements.
If you want to experiment with your data, make sure to follow the guidelines in the *About ml4bio* page.

## Machine learning workflow (Make our own version of this)

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/tcells1.jpeg">
</p>

### Data preprocessing 

The first step in machine learning is to prepare our data.
__Preprocessing__ the raw data is an essential step to have quality data for the model.


> ## Definitions
>
> Preprocessing - text
>
> Missing values - text
>
> Outliers - text
>
> Data normalization - text
{: .callout}

Preprocessing data can include imputing __missing values__, checking the consistency of the data's features, choosing how to deal with any __outliers__, removing duplicate values, and converting all features into a format that is usable by a machine learning algorithm.
Ther are a variety of methods and tools for data __normalization__ and preprocessing.

However, learning these methods and tools is outside the scope of this workshop because as preprocessing strategies are specific to both a dataset's domain and the technology used to gather the data.
Throughout this workshop we will assume that all of the data has already been preprocessed. 

> ## Software
>
> Load size_intensity_feature.csv into the software under the Labeled Data.
{: .checklist}

> ## Conceptual Questions
>
> What are we trying to predict? 
>
> What features do we have for those predictions?
{: .challenge}

## Step 1: Select data

### Data summary

Data Summary gives us an insight into features and __samples__ for the dataset we selected.

> ## Definition
>
> Sample - text
{: .callout}

In this particular dataset, we can see that we have two features **cell_size** and **total_intensity**.
We can also see that the total number of samples is 843. 

> ## Conceptual Questions
>
> How many quiescent samples are in the dataset?
>
> How many active samples?
> 
> Can we make any assumptions about the dataset based on the number of samples for each label?
{: .challenge}

### Training set vs. Validation set vs. Test set 

Before we continue, we need to split the dataset into a training set and a test set.
The training set is further divided into a training set and a validation set. 

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/tcells2.jpg">
</p>

> ## Definitions
>
> Training set - The training set is a part of the original dataset that trains or fits the model. This is the data that the model uses to learn.
>
> Test set - The test set checks how well the model we expect the model to work on new data in the future. The test set is used in the final phase of the workflow, and it evaluates the final model. 
>
> Validation set - Further, part of the training set is used to validate that the fitted model works on new data.
> This is not the final evaluation of the model.
> This step is used to change __hyperparameters__ and then train the model again. _Comment: Define hyperparams_
>
> Hyperparameters - These are the settings of a machine learning model. Each machine learning method has different hyperparameters, and they control various trade-offs which change how the model learns. 
> Hyperparameters control parts of a machine learning method such as how much emphasis the method should place on being perfectly correct versus becoming overly complex, how fast the method should learn, the type of mathematical model the method should use for learning, and more.
{: .callout}

Setting a test set aside from the training and validation sets from the beginning, and only using it once for a final evaluation, is very important to be able to properly evaluate how well a machine learning algorithm learned.
Letting the machine learning method learn from the test set can be seen as giving a student the answers to an exam; once a student sees any exam answers, their exam score will nol longer reflect their performance in the class.  

We will be using the holdout validation method in the software.
We will use the software's default of 20% of the training set for the validation set.
<p align="center">
<img width="700" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/tcells3.jpg">
</p>


## Step 2: Train classifiers

__Classification__ is a process when given some input we are try to predict an outcome by coming up with a rule that will guess this outcome.
Tools that use classification to predict an outcome are __classifiers__.

> ## Definitions
>
> Classification - text
>
> Classifier - text
{: .callout}


The software has a dropdown menu of some of the most frequently used classifiers.
In this workshop, we will be further talking about [Decision Trees](https://gitter-lab.github.io/ml-bio-workshop/03-decision-trees/index.html), [Random Forests](https://gitter-lab.github.io/ml-bio-workshop/04-random-forests/index.html), [Logistic Regression](https://gitter-lab.github.io/ml-bio-workshop/05-log-regression/index.html), and [Neural Network](https://gitter-lab.github.io/ml-bio-workshop/07-neural-nets/index.html).

We will evaluate the classifiers using accuracy.Accuracy measures the fraction or the count of the correct predictions. In the T-cells dataset, this will be the number of correctly predicted quiescent and activated cells compared to the total number of predictions made. 
In the software, let's look at the prediction metrics on the validation data.
Remember, you can switch between the training set and validation set at any time in software. 
In the T-cells example, we want to predict whether a cell was quiescent or activated. The accuracy gives us the count of the cells that were correctly predicted.

> ## Exploring model training
>
> Pick a few classifiers and train them.
>
> Try to answer these questions to get a better understanding of the software:
> 
> How does changing the validation set percentage change the training set and validation set accuracies?
>
> How do training set accuracy and validation set accuracy tend to compare to each other? Why do you think this is?
>
> What is the highest validation set accuracy you can get a classifier to achieve?
{: .checklist}

This type of exploration of multiple algorithms reflects how a good model is often found in real-world situations.
It often takes many classifiers before you find the one that you are satisfied with.

For your final comparison, train at least one decision tree, random forest, logistic regression and logistic regression classifier.


## Step 3 Test and Predict

Our final step is model selection and evaluation.
After we trained multiple classifiers and did holdout validation, the next step is to choose the best model.

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/Screen%20Shot%202019-11-08%20at%2010.40.25%20AM.png">
</p>

There are a few visualization tools that can help with the model selection. The __Confusion Matrix__ reflects the selected dataset (training or validation). The T-cells dataset has two labels so that the Confusion Matrix will be 2 x 2. The sum of all the predictions will be the total number of samples in the selected dataset (training or validation). 

> ## Definitions
>
> Confusion matrix - text
{: .callout}

### Test Data

Once the model was selected based on the metric that we chose, we want to use the model for the prediction on the test data.
Based on the same prediction metrics that we used on the validation set, we can make certain conclusions about our model. 

> ## Conceptual Questions
>
> How did your final test set accuracy compare to your validation accuracy?
{: .challenge}

#### Image attributions

> The T-cell images come from [Wang et al. 2019](https://doi.org/10.1002/jbio.201960050) with data originally from [Walsh et al. 2019](https://doi.org/10.1101/536813).

{% include links.md %}
