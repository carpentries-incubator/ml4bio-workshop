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
---

### Why classify T cells

Let's consider how to answer a real biological question using the same concepts we saw in the introduction.
The question pertains to immunotherapy, a type of cancer treatment that uses the body's immune cells to boost natural defenses against cancer.
T cells are a common target for immunotherapies.
For immunotherapy to be effective, the modified T cells must be in an active state.
Here we will study how to assess the activation state of individual T cells.

Scientists at UW-Madison and the Morgridge Institute [developed an imaging method](https://doi.org/10.1101/536813) to easily and quickly acquire images of T cell without destroying them.
These images contain information that can be used to predict T cell activity.
We would like to develop a classifier that can take an image of a T cell and predict whether it is active or quiescent.
The active cells would then be used for immunotherapy, and the quiescent cells can be considered inactive and would be discarded.
_Comment: Make sure we include Jay in acknowledgements when we update them_
_Comment: Define "active" in the sentence above_

### Dataset description

This microscopy dataset includes grayscale images of two type of T cells: activated and quiescent.
These T cells come from blood samples from six human donors.
_Comment: Do we need to make these images bigger? Does that require HTML images? Add another row or two so we can see some patterns emerge (Tony)._

|Activated|Quiescent|
|:---:|:---:|
|![CD3_2_act_2_6_66](https://user-images.githubusercontent.com/15007159/61666368-e4804d00-ac9c-11e9-9031-a3f9f6cfd7b1.png)|![CD8_2_noact_3_3_13](https://user-images.githubusercontent.com/15007159/61666346-d9c5b800-ac9c-11e9-9044-e13c218d0da0.png)|

We will use a subset of the images and follow the workflow in a [T cell classification study](https://doi.org/10.1002/jbio.201960050).

_Comment: Restructure to be like slides, add 2-3 examples, ask questions about what we think the labels are (show of hands)_

### Machine learning methods

The goal of this study is to develop a method to classify T cell activation state (activated vs. quiescent). 

### ml4bio software setup

We will be using ml4bio software to build classifiers with the T cell images.
Refer to the [Setup](https://gitter-lab.github.io/ml-bio-workshop/setup.html) for instructions on how to install and launch the ml4bio software. _Comment: We should learn how to use relative links_ 
To better understand the software features, check out the [About ml4bio](https://gitter-lab.github.io/ml-bio-workshop/about/index.html) page.
All of the datasets that we will be using for this workshop have been formatted to fit the ml4bio requirements.
If you want to experiment with your data, make sure to follow the guidelines in the *About ml4bio* page.

## Machine learning workflow (Make our own version of this)

_Comment: If we reuse, crop the bubbles out and add attribution at the bottom of the page._
<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/workflow.png">
</p>

### Data preprocessing 

The first step in machine learning is to prepare our data.
Preprocessing the raw data is an essential step to have quality data for the model.
Some of the properties of quality data are the absence of missing values, the data for each feature is of consistent data type and the same unit of measure, any outliers have been addressed, and there are no duplicate values.
Some methods and tools are used for data normalization and preprocessing.
However, learning these methods and tools is not one of the objectives of the workshop because of the time constraint and the focus on classification and choosing a model.
So, we will assume that all of the data has already been preprocessed.
_Comment: missing values, outliers need to be defined.  Leave this?  Make bullets?_

**Load size_intensity_feature.csv into the software under the Labeled Data**
_Comment: This should be some type of callout box.  Make steps to be performed in ml4bio software into .checklist style callouts._

> ## Conceptual Questions
>
> What are we trying to predict? 
>
> What features do we have for those predictions?
{: .challenge}

## Step 1: Select data

### Data summary

Data Summary gives us an insight into Features and Samples for the dataset we selected.
In this particular dataset, we can see that we have two features **cell_size** and **total_intensity**.
We can also see that the total number of Samples is 843. 

> ## Conceptual Questions
>
> How many quiescent samples are in the dataset?
>
> How many active samples?
> 
> Can we make any assumptions about the dataset based on the number of samples for each label?
{: .challenge}

### Training set vs. Validation set vs. Test set 

The preprocessed dataset is split into a training set and a test set.
The training set is further divided into a training set and a validation set. 

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets.jpg">
</p>

_Comment: Use new definition styles for these terms.  Calling this approach Holdout Validation, which also needs to be defined here._
#### Training set 

The training set is a part of the original dataset that trains or fits the model.
This is the data that the model uses to learn.

#### Validation set

Further, part of the training set is used to validate that the fitted model works on new data.
This is not the final evaluation of the model.
This step is used to change hyperparameters and then train the model again. _Comment: Define hyperparams_
We will be using the holdout validation method in the software.
We will use the ml4bio default 20% of the training set for the validation set.

What is commonly done in practice is cross-validation.
One part of the training set is used for training and another section for validation.
Then the hyperparameters of the fitted models are changed, and the model has trained again with the new split between training data and the validation data.
Then we look through all folds of data.
The same is done with the training dataset.
Each time we repeat training and validating, we split the original training set into new training and validation datasets.

_Comment: Replace Cross-Validation with Model Tuning and remove the red "repeat splitting" bubble. Cross validation would be a Day 2 workshop concept. _
<p align="center">
<img width="700" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets2.jpg">
</p>


#### Test set

The test set checks how well the model we expect the model to work on new data in the future.
The test set is used in the final phase of the workflow, and it evaluates the final model. 
_Comment: Say something about only using the test set once._

We will use the ml4bio default option and splitting our dataset to be 80% training set and 20% test set. 

## Step 2: Train classifiers

Classification is a process when given some input we are try to predict an outcome by coming up with a rule that will guess this outcome.
Tools that use classification to predict an outcome are classifiers.
_Comment: Callout for definition_

We are given a dropdown menu of some of the most frequently used classifiers in biology.
In this workshop, we will be further talking about [Decision Tree](https://gitter-lab.github.io/ml-bio-workshop/03-decision-trees/index.html), [Random Forests](https://gitter-lab.github.io/ml-bio-workshop/04-random-forests/index.html), [Logistic Regression](https://gitter-lab.github.io/ml-bio-workshop/05-log-regression/index.html), and [Neural Network](https://gitter-lab.github.io/ml-bio-workshop/07-neural-nets/index.html).

We will evaluate the classifiers using accuracy.
_Comment: Define accuracy_

> ## Play time
>
> Pick a few classifiers and train them without changing the default hyperparameters.
>
> Do you see different accuracy depending on the classifier.
{: .checklist}
_Comment: Do we need more instructions about how to do this in the software?  Or will we show them?_

_Comment: Clean up text and move some into the box above_
As you can see, you will get different performance metrics depending on the classifier. _Comment: Tell them where the metrics are_
This reflects the real life situtation when you work with the real data.
You will train many classifiers before you find the one that you are satisfied with.

Try to answer these questions to get a better understanding of the software:
- (FIX THIS)

After you finish playing around, let's train Decision Tree, Random Forest, Logistic regression, and Neural Network classifiers. 

For this specific example, we will be working with the default hyperparameters in all cases.


## Step 3 Test and Predict

Our final step is model selection.
After we trained multiple classifiers, changed some hyperparameters, and did holdout validation, the next step is to choose the best model.
Model evaluation and selection is a vast topic so that we will be focusing on the metrics provided in the software.

_Comment: Recommend focusing on 1 or maybe 2 metrics here (maybe accuracy?).  We want to show the end-to-end workflow and not get stuck explaining all the metrics.  Those can come is the choosing a model episode or elsewhere._
- Accuracy measures the fraction or the count of the correct predictions. In the T-cells dataset, this will be the number of correctly predicted quiescent and activated cells compared to the total number of predictions made. In the software, let's look at the prediction metrics on the validation data. Remember, you can switch between the training set and validation set at any time in software. In the T-cells example, we want to predict whether a cell was quiescent or activated. The accuracy gives us the count of the cells that were correctly predicted. In this section of the software

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/Screen%20Shot%202019-11-08%20at%2010.40.25%20AM.png">
</p>

there are a few visualization tools that can help with the model selection. The *Confusion Matrix* reflects the selected dataset(training or validation). The T-cells dataset has two labels so that the Confusion Matrix will be 2 x 2. The sum of all the predictions will be the total number of samples in the selected dataset(training or validation). 

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/Screen%20Shot%202019-11-08%20at%2010.40.25%20AM.png">
</p>

> ## Conceptual Questions
>
> Explain the meaning of the Confusion Matrix to a partner.  
{: .challenge}

- Precision is the proportion of the cells correctly predicted as activated compared to the total number of cells predicted as activated regardless if they were correctly predicted or not. 
- Recall is the proportion of the cells correctly predicted as activated compared to the sum of those cells correctly predicted as activated and those incorrectly predicted as quiescent. Although the Precision-Recall curve visualized these two metrics, we will not be going into details about it in this version of the workshop. 
- F1 
- AUROC
- AUPRC


### Test Data

Once the model was selected based on the metric that we chose, we want to use the model for the prediction on the test data.
Based on the same prediction metrics that we used on the validation set, we can make certain conclusions about our model. 

_Comment: Add callout to ask if whoever had the best validation accuracy also has the best test set accuracy.  Why or why not is this the case?_

_Comment: Should summarize the key points and add them above in the Markdown header.  What can we now do with new T cell images?_

{% include links.md %}
