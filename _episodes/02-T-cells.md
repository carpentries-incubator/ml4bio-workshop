---
title: "T-cells"
teaching: 30
questions:
- "What are we trying to predict?"
objectives:
- "Comprehend how the datasets are preprocessed, what are samples, features and class labels, what is a model in machine learning, what is training set, hyperparameters, validation set, evaluation and prediction, and how to perform model selection."
---

Immunotherapy is a type of cancer treatment that uses the body’s own immune cells to boost natural defenses against cancer, and T cells are a popular target for immunotherapies. To fully optimize the effect of immunotherapy, T cell activation must be assessed at a single-cell level. Melissa Skala's group developed a label-free and non-destructive autofluorescence imaging method to easily and quickly acquire T cell intensity images. Therefore, an activated/quiescent (inactive) T cell classifier that uses autofluorescence intensity images can contribute to the applications of immunotherapy.

### Dataset description

This microscopy dataset includes gray scale images of two type of T cells: activated and quiescent (not activated). These T cells come from 6 donors.

|Activated|Quiescent|
|:---:|:---:|
|![CD3_2_act_2_6_66](https://user-images.githubusercontent.com/15007159/61666368-e4804d00-ac9c-11e9-9031-a3f9f6cfd7b1.png)|![CD8_2_noact_3_3_13](https://user-images.githubusercontent.com/15007159/61666346-d9c5b800-ac9c-11e9-9044-e13c218d0da0.png)|

Since WARF team is filing a patent for this study, we can only make these subsampled images public.

||Donor 1|Donor 2|Donor 3|Donor 4|Donor 5|Donor 6|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Full Activated|271|656|487|494|694|446|
|Full Quiescent|1631|152|1276|1629|252|589|
|Subsampled Activated|27|65|48|49|69|44|
|Subsampled Quiescent|163|15|127|162|25|58|

### Machine Learning Methods

The goal of this study is to develop a method to classify T cell activation stage (activated vs. quiescent). 

### ML4Bio software setup

We will be using ML4 Bio software. Refer to the [Setup](https://gitter-lab.github.io/ml-bio-workshop/setup.html) for instructions on how to install and launch the software. 

## Machine Learning workflow (Make our own version of this)

![workflow](https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/workflow.png "Figure from 
S. Raschka and V. Mirjalili, Python Machine Learning (2nd Ed.)")

### Data Preprocessing 

The first step in machine learning would be to prepare our data. Preprocessing the raw data is an important step in order to have quality data for the model. There are methods and tools that are used for data normalization and preprocessing. However, learning these methods and tools is not one of the objectives of the workshop. So, we will assume that all of the data has already been preprocessed. 

**Load size_intensity_feature.csv into the software under the Labeled Data**

> ## Conceptual Questions
>
> What are we trying to predict? 
> What features do we have for those predictions?
{: .challenge}

## Step 1 Select Data

### Data Summary

Data Summary gives us an insight into Features and Samples for the dataset we selected. In this particular dataset, we can see that we have two features **cell_size** and **total_intensity**. We can also see that the total number of Samples is 843. 

> ## Conceptual Questions
>
> How many quiescent samples are in the dataset? How many activated? 
>Can we make any assumptions about the dataset based on the number of samples for each label?
{: .challenge}

### Training set vs. Validation set vs. Test set 

The preprocessed dataset is split into a training set and a test set. The training set is further split into a training set and a validation set. 
![datasets](https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets.jpg)

#### Training set 

The training set is a part of the original dataset that trains or fits the model. This is the data that the model uses to learn.

#### Validation set

Further, a part of the training set is used for validation of the fitted model. This is not the final evaluation of the model. This step is used to change hyperparameters and then train the model again.  We will be using the holdout validation method in the software. We will use the default 20% of the training set for the validation set.

What is commonly done in practice is cross validation. One part of the training set is used for training, and another section for validation. Then the hyperparameters of the fitted models are changed, and the model has trained again with the new split between training data and the validation data. A good example to explain this process would be dealing the cards. Every time we deal with the cards, we shuffle them first. The same is done with the training dataset. Each time we repeat training and validating, we split the original training set into new training and validation datasets. 

![datasets2](https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets2.jpg)

#### Test set

 The test set checks how well the model works on the new data. The test set is used in the final phase of the workflow, and it evaluates the final model. 

In the GUI we will be using the default option and splitting are dataset to be 80% training set and 20% test set. 

### Validation methodology

#### Holdout Validation

#### K-Fold Cross Validation

#### Leave One-Out Validation

#### Stratified Sampling check box

## Step 2 Traing Classifiers

We are given a dropdown menu of some of the most frequently used classifiers in biology. In this workshop, we will be further talking about Decision Tree, Random Forest, Logistic Regression and Neural Network. 

> ## Play time
>
> Pick any classifier and not changing the default settings train the data. 
{: .callout}





