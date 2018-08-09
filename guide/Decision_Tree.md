# Decision Tree

###Representation

<p align="center">
<img src="../figures/decision_tree/rep.jpg">
</p>

###Learning

Denote the set of features by ![](../figures/decision_tree/features.gif)
and the set of classes by ![](../figures/decision_tree/classes.gif).
Starting with the full training data ![](../figures/decision_tree/D.gif) of size ![](../figures/decision_tree/N.gif), 
we grow a tree by splitting data at each node into two parts according to the values of a selected feature, 
until no more split can be generated (or some regularization techniques are applied). 

Conceptually, we want to split the data at a node such that the division of class labels becomes clearer at the two children nodes. 
Technically, we choose the most _informative_ feature at each node, 
and the split results in the biggest reduction in _uncertainty_.

The math below defines the measure of _uncertainty_ and specifies what we exactly mean by an _informative_ feature.

Given the training data ![](../figures/decision_tree/D_m.gif) of size ![](../figures/decision_tree/N_m.gif) at node ![](../figures/decision_tree/m.gif),
the frequency of samples in class ![](../figures/decision_tree/k.gif) is

<p align="center">
<img src="../figures/decision_tree/learning_eq_1.gif">
</p>

where ![](../figures/decision_tree/x.gif) is a sample in ![](../figures/decision_tree/D_m.gif) and y is its label.

There are two common measures of uncertainty (or impurity):

<p align="center">
<img src="../figures/decision_tree/uncertainty.jpg">
</p>

To understand these measures, we consider a binary problem with classes 0 and 1, 
and plot gini score and entropy against ![](../figures/decision_tree/p_k.gif).

<p align="center">
<img width=500 src="../figures/decision_tree/uncertainty_fig.png">
</p>

It is easy to see that both measures reach the maximum when ![](../figures/decision_tree/p_k_half.gif), which agrees with our intuition.

Given feature ![](../figures/decision_tree/f_i.gif),

<p align="center">
<img src="../figures/decision_tree/learning_eq_2.gif">
</p>

where

<p align="center">
<img src="../figures/decision_tree/learning_eq_3.gif">
</p>

Split with the feature ![](../figures/decision_tree/f_star.gif) that reduces uncertainty the most:

<p align="center">
<img src="../figures/decision_tree/learning_eq_4.gif">
</p>

To avoid overfitting, we may tune the following hyperparameters to regularize decision tree learning.

- the maximum depth of the tree
- the minimum number of samples required to split a node
- the minimum number of samples required to be at a leaf node
- the maximum number of leaf nodes
- the minimum decrease in uncertainty required for learning to continue
- the minumum uncertainty required for splitting a node

###Inference

Given a new sample ![](../figures/decision_tree/x.gif), 
move down the tree by choosing the right path at each split.
Finally, you end up in a leaf node that tells you which class the sample belongs to.

###Example

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/tree.html) on Decision Trees.
> 2. sklearn `DecisionTreeClassifier` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).
