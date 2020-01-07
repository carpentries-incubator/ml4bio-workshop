---
title: "T-cells"
teaching: 30
questions:
- "What are we trying to predict?"
objectives:
- "Comprehend how the datasets are preprocessed, what are samples, features and class labels, what is a model in machine learning, what is training set, hyperparameters, validation set, evaluation and prediction, and how to perform model selection."
---

Let's consider how to answer a real biological question using the same concepts we saw in the introduction.
The question pertains to immunotherapy, a type of cancer treatment that uses the body's immune cells to boost natural defenses against cancer.
T cells are a common target for immunotherapies.
For immunotherapy to be effective, the modified T cells must be in an active state.
Here we will study how to assess the activation state of individual T cells.

Scientists at UW-Madison and the Morgridge Institute [developed an imaging method](https://doi.org/10.1101/536813) to easily and quickly acquire images of T cell without destroying them.
These images contain information that can be used to predict T cell activity.
We would like to develop a classifier that can take an image of a T cell and predict whether it is active or inactive.
The active cells could them be used for immunotherapy, and the inactive cells could be discarded.
_Comment: Is this our text or was it derived from another source?  Need to avoid as much bio jargon as possible and use inactive instead of quiescent._

### Dataset description

This microscopy dataset includes grayscale images of two type of T cells: activated and inactive.
These T cells come from blood samples from six human donors.
_Comment: Do we need to make these images bigger?_

|Activated|Quiescent|
|:---:|:---:|
|![CD3_2_act_2_6_66](https://user-images.githubusercontent.com/15007159/61666368-e4804d00-ac9c-11e9-9031-a3f9f6cfd7b1.png)|![CD8_2_noact_3_3_13](https://user-images.githubusercontent.com/15007159/61666346-d9c5b800-ac9c-11e9-9044-e13c218d0da0.png)|

We will use a subset of the images and follow the workflow in a [T cell classification study](https://doi.org/10.1002/jbio.201960050).

_Comment: Can we remove this table?  What are we trying to convey here?_
The table presents the number of cells for each donor. 

||Donor 1|Donor 2|Donor 3|Donor 4|Donor 5|Donor 6|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Full Activated|271|656|487|494|694|446|
|Full Quiescent|1631|152|1276|1629|252|589|
|Subsampled Activated|27|65|48|49|69|44|
|Subsampled Quiescent|163|15|127|162|25|58|

### Machine learning methods

The goal of this study is to develop a method to classify T cell activation stage (activated vs. quiescent). 

### ml4bio software setup

We will be using ml4bio software to build classifiers with the T cell images.
Refer to the [Setup](https://gitter-lab.github.io/ml-bio-workshop/setup.html) for instructions on how to install and launch the ml4bio software. _Comment: We should learn how to use relative links_ 
To better understand the software features, check out the [About ml4bio](https://gitter-lab.github.io/ml-bio-workshop/about/index.html) page.
All of the datasets that we will be using for this workshop have been formatted to fit the ml4bio requirements.
If you want to experiment with your data, make sure to follow the guidelines in the *About ml4bio* page.

## Machine learning workflow (Make our own version of this)

_Comment: Are we going to reuse this image?_
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

**Load size_intensity_feature.csv into the software under the Labeled Data**
_Comment: This should be some type of callout box_

> ## Conceptual Questions
>
> What are we trying to predict? 
> What features do we have for those predictions?
{: .challenge}

## Step 1: Select data

### Data summary

Data Summary gives us an insight into Features and Samples for the dataset we selected.
In this particular dataset, we can see that we have two features **cell_size** and **total_intensity**.
We can also see that the total number of Samples is 843. 

> ## Conceptual Questions
>
> How many inactive samples are in the dataset? How many active? 
> Can we make any assumptions about the dataset based on the number of samples for each label?
{: .challenge}

### Training set vs. Validation set vs. Test set 

The preprocessed dataset is split into a training set and a test set.
The training set is further divided into a training set and a validation set. 

<p align="center">
<img width="600" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets.jpg">
</p>

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
An example to explain this process would be dealing playing cards.
Every time we deal the cards, we shuffle them first.
The same is done with the training dataset.
Each time we repeat training and validating, we split the original training set into new training and validation datasets.
_Comment: If we loop through all folds of the data, that isn't exactly following the cards example._

<p align="center">
<img width="700" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/datasets2.jpg">
</p>


#### Test set

The test set checks how well the model we expect the model to work on new data in the future.
The test set is used in the final phase of the workflow, and it evaluates the final model. 
_Comment: Say something about only using the test set once._

We will use the ml4bio default option and splitting our dataset to be 80% training set and 20% test set. 

### Validation methodology

_Comment: Do we have time to discuss all of these or should we focus only on holdout validation?_
Now that we learned about the importance of splitting our dataset, let's briefly mention some of the validation methodologies. 

#### Holdout Validation

#### K-Fold Cross Validation

tried to write about this, but it is hard to be concise. Do we want this? Or just to tell them which one to use in the January workshop? 


## Step 2: Train classifiers

We are given a dropdown menu of some of the most frequently used classifiers in biology.
In this workshop, we will be further talking about Decision Tree, Random Forest, Logistic Regression, and Neural Network. _Comment: Can we link to the episodes?_

> ## Play time
>
> Pick a few classifiers and without changing the default settings train the data.
{: .callout}
_Comment: Do we need more instructions about how to do this in the software?  Or will we show them?_

As you can see, you will get different performance metrics depending on the classifier. _Comment: Tell them where the metrics are_
This reflects the real life situtation when you work with the real data.
You will train many classifiers before you find the one that you are satisfied with.

Try to answer these questions to get a better understanding of the software:
- (FIX THIS)

After you finish playing around, let's train Decision Tree, Random Forest, Logistic regression, and Neural Network classifiers. 

For this specific example, we will be working with the default hyperparameters in call cases.


## Step 3 Test and Predict

Our final step is model selection.
After we trained multiple classifiers, changed some hyperparameters, and did cross-validation, the next step is to choose the best model.
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
_Comment: Should summary the key points.  What can we now do with new T cell images?_

> ## Key points
>
> 
{: .keypoints}
