---
title: "Instructor Notes"
---
## Tips
When instructing, repeat 3 times:
- talk about the process 
- show how it is done
- instruct the participants to do it themselves

Give clear indicators of when learners should be looking at the projector screen, the instructor, or their own laptop (or combination).

## Introduction
Initial think-pair-share on machine learning is very short.

For the housing example, the instructor will recreate the table on the board and collect the data from the participants.
Write the answers on the board.
Provide one example feature, then open the discussion for others.
Select two features that will work well for the classification example.

## T cells
When training classifiers, train one type of classifier together.
Point out where the accuracy is in the table and how it is derived from the confusion matrix.
Then have participants try more classifiers.
Ask the participants who obtained the best accuracy and what settings they used.

## Decision Trees
The house price example is more like a regression example than classification, but we use it because it is very intuitive.


## Random Forests

### Overfitting Cont. 

This introduction is intended to motivate random forests and introduce some fondational ideas in learning theory. The goal of this is to make participants begin to consider _why_ a model succeeds, and how data scientists think about machine learning. 

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


{% include links.md %}
