---
title: "Random Forests"
teaching: 30
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

_Comment: Need to add Overview info_

Random forest is a classifier that combines a large number of decision trees.
Each decision tree in the forest makes a prediction, and the final prediction is the class label that the most decision trees predicted.
Combining many different decision trees makes the final prediction more robust and less prone to overfitting.
_Comment: Say more about the RF algortihm and how it randomlly selects features and instances to train on_

**How do we choose a classifier?** (finish)

### Step 1 Select data

Let's load a T-cells #3 data set into our software. _Comment: State the exact filename_
This dataset is designed specifically illustrate properties of the random forest classifier.

> ## Conceptual Questions
>
> What are we trying to predict? 
> What is the decision rule?
> How would the random forest look graphically?
{: .challenge}

_Comment: Need to fill in the rest of these terms and add step-by-step instructions of what to do in the software_

Sample answer for the visual


### Step 2 Train Classifiers

In this workshop not all of the hyperparameters from the software will be covered.
Those that we don't cover, we will be using the default settings. 
- Skip criterion
- N-estimator - # decision trees that go into forest
- Max-features not so important
- Bootstrap not essential 


### Step 3 Test and Predict

**Finish once the dataset is done**

###  Application in biology

