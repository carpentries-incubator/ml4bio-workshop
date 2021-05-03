---
title: "Instructor notes"
---
## Tips
When instructing, repeat 3 times:
- talk about the process 
- show how it is done
- instruct the participants to do it themselves

Give clear indicators of when learners should be looking at the projector screen, the instructor, or their own laptop (or combination).

## Introduction
After warm-up, go through the 3 ML examples:


1. You are trying to understand how temperature affects the speed of embryo development in mice. After running an experiment where you record developmental milestones in mice at various temperatures, you run a linear regression on the results to see what the overall trend is. You use the regression results to predict how long certain developmental milestones will take at temperatures you’ve not tested.

    *This is machine learning*
    *There is a model, the linear regression, which is learning from instances and being used for prediction.*
    *Even though a linear regression is simple, when used in this way it is machine learning*

2. You want to create a guide for which statistical test should be used in biological experiments. You hand-write a decision tree based on your own knowledge of statistical tests. You create an electronic version of the decision tree which takes in features of an experiment and outputs a recommended statistical test.

    *While this example contains a decision tree, which is used as a classifier, there is no learning from data.*
    *The decision tree is instead being created by hand.*
    *This is not machine learning.*

3. You are annoyed when your phone rings out loud, and decide to try to teach it to only use silent mode. Whenever it rings, you throw the phone at the floor. Eventually, it stops ringing. “It has learned. This is machine learning,” you think to yourself.

    *This example appears to contain instances and change with experience, but lacks a model.*
    *There is no way to apply anything to future data.*
    *This is not machine learning.*

## T cells
Data leakage scenarios:

1. We created a decision tree model to predict whether the compound would inhibit cell growth. We trained the model on the 48 available instances, and found that the decision tree was able to predict those instances with an accuracy of 0.96. Thus, the decision tree is high performing on this task.

    *This is the improper usage of data.*
    *Clues that can lead us to this conclusion are that there is no mention of splitting the data between a training set and a testing set.*
    *The only information that is provided is that the accuracy was on the same instances that the model was trained with.*
    *This accuracy value is not representative of what would be gotten with new data.*

2. We trained 36 different models, each using a different combination of hyperparameters. We trained each model on 80% of the data, withholding 20% of the data to test each model. We present the highest performing model here to show the effectiveness of machine learning on this task. 

    *In this example, we do see that there is a split between the training set and the testing set.*
    *However, the testing set was used to select the best of the 36 models instead of a validation set.*
    *Thus, information from the testing set has __leaked__ into creating the model.*
    *The models performance is thus likely inflated.*

3. We split the data into training and testing sets of 80% and 20%, and further split the training set into a training and validation set. We trained 200 models on the training data, and chose the best-performing model based on performance on the validation set. After choosing and training the model, we found that the model had an accuracy of 0.93 on the testing set.

    *This example does not appear to have any data leakage.*
    *The data was properly split into a training and validation set, and the best model was chosen without looking at the testing set.*
    *Thus, we can trust that the performance on the testing set would represent performance on new data given to the model.*

## Evaluating a model


## Trees and overfitting

Scenarios:

In the following scenarios, which classifier would you choose?

You want to create a model to classify a protein’s subcellular localization (nucleus, mitochondria, plasma membrane, etc.). You have a labeled set of 15,000 human proteins with 237 features for each protein. These features were computationally derived from simulations using protein structure predicting software, and do not have any predefined meaning.

*In this scenario, we would probably want to use a random forest.*
*There are a lot of features and data, which means that a random forest's increased complexity is a good thing here.*
*The features are also not that interpretable, so we're not too concerned with being able to understand why the model is making certain decisions.*

You want to create a model to predict whether or not a species’ conservation status (least concern, endangered, extinct, etc.) will be affected by climate change. You have a labeled dataset of 40 species, with 18 features for each species which have been curated by ecologists. These features include information such as the species’ average size, diet, taxonomic class, migratory pattern, and habitat. You are interested to see which features are most important for predicting a species’ fate.

*We have less data in this scenario, so a simpler model might be better.*
*Decision trees are likely the right choice here, especially since we have a very informative, small set of features.*
*We are also interested in being able to interpret the model, which decision trees excel at and random forests can struggle with.*

### Overfitting Cont. 


__Only do this part if we are ahead of schedule.__

This introduction is intended to motivate random forests and introduce some foundational ideas in learning theory. The goal of this is to make participants begin to consider _why_ a model succeeds, and how data scientists think about machine learning. 

First re-hash overfitting, pointing out the definitions of bias and variance. 
    Emphasize: high bias -> simpler models less subject to change -> less overfitting, more training error
    high variance -> more complex models more subject to change -> more overfitting, less training error


Now we're going to draw another plot to highlight the relationship between overfitting and the amount of training data
<p align="center">
<img width="900" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/randomForestsOverfitting.png">
</p>

`First draw the data plot on the left. `
Imagine that it is a st of data for a drug response.
Ask what looks right as a fit line, the straight line or curvy line?
The straight line feels better, as those curves are making a lot of assumptions on the sape of the dose response based on only one data point. 

`Now add data points so that the plot looks like the plots on the right.` 
With additional data, we the curvy line looks more appropriate.
`Write out the equation for the straight line, y=mx+b (high bias, low variance)`
`Now write the beginnings of a high-order polynomial for the curvy line, y=ax^9+bx^7+... (low bias, high variance)`

Explain how this shows that more data allows us to support more complex models, with a deeper connection to how many _parameters_ the model has. 

### Random forest motivation

Now that we understand the basics of the bias variance trade-off, we can look at how random forests try to get around it.

`Draw the left plot with fewer dots again, and the curvy high variance line`

The problem with high variance models is that we're learning too much to random variation in the training set, new data we collect probably won't follow the same pattern.

But what if we had a bunch of small training sets, created high-variance models, then averaged them together?

`Draw multiple curvy lines with different shapes on the plot`

We could average out the error caused by variance accross the models, while still maintaining the flexibility and low bias these models give us. 

This is the intuition behind random forests, where we use a many trees [a forest of trees]  which we setup to have high variance and use them together to choose a final classification. 

However, there is still a problem.
How do we get enough data for all of these high-variance models?
This is where the random in random forests comes in.

We fake having more data by randomly sampling with replacement from the training data. 
In order to further make sure that our curvy lines are different enough, we also only use a subset of all the features, also chosen randomly, for each tree. 

## Logistic Regression, Artificial Neural Networks, and Linear Separability

### Artificial neural networks in practice

We're not expected to be able to understand everything ion the robotic surgery arm example. 
However, we can see that in a modern deep neural network the building blocks are still then same as the simpler models we've looked at. 
We have additional layers which are performing complex tasks on the features of the data to create rich features the final classification layers can predict on. 

### Classifier selection scenarios

In the following scenarios, which classifier would you choose?

You are interested in learning about how different factors contribute to different water preservation adaptations in plants. You plan to create a model for each of 4 moisture preservation adaptations, and use a dataset of 200 plant species to train each model. You have 15 features for each species, consisting of environmental information such as latitude, average temperature, average rainfall, average sun intensity, etc.

*While we can't be sure, it is likely that each of these features is linear with the class of the data.*
*Things like temperature, rainfall, etc. probably affect moisture preservation adaptations in a single direction.*
*We're also trying to learn about how factors contribute, not make perfect classifications, so interpretability is important.*
*Therefore, logistic regression is probably the best choice.*

You have been tasked with creating a model to predict whether a mole sample is benign or malignant based on gene expression data. Your dataset is a set of 380 skin samples, each of which has expression data for 50 genes believed to be involved in melanoma. It is likely that a combination of genes is required for a mole to be cancerous.

*Since it likely takes combinations of mutations to cause a sample to be malignant, the data is probably non-linear.*
*We wouldn't want to treat the 50 genes independently.*
*Therefore, we should use a neural network for this task.*

## Understanding machine learning literature

## Conclusion

{% include links.md %}
