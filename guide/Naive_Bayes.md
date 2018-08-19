# Naïve Bayes

### Representation

<p align="center">
<img src="../figures/naive_bayes/rep_fig.jpg">
</p>

<p align="center">
<img src="../figures/naive_bayes/rep_formula_1.jpg">
<img src="../figures/naive_bayes/rep_formula_2.jpg">
</p>



### Learning

Denote the set of features by ![](../figures/naive_bayes/features.gif), 
and the set of classes by ![](../figures/naive_bayes/classes.gif).

Given the full training data ![](../figures/naive_bayes/D.gif) of size ![](../figures/naive_bayes/N.gif), 
let ![](../figures/naive_bayes/N_k.gif) be the number of samples that belong to class ![](../figures/naive_bayes/k.gif).
We estimate the frequency of each class by

<p align="center">
<img src="../figures/naive_bayes/learning_eq_1.gif">
</p>

For a discrete feature ![](../figures/naive_bayes/f_j.gif),

<p align="center">
<img src="../figures/naive_bayes/learning_eq_2.gif">
</p>

where ![](../figures/naive_bayes/alpha.gif) is the **smoothing prior** 
and ![](../figures/naive_bayes/mj.gif) is the total number of 
values of ![](../figures/naive_bayes/f_j.gif).

For a continuous feature ![](../figures/naive_bayes/f_j.gif),
the parameters of ![](../figures/naive_bayes/cond_dist.gif) 
are estimated by maximum likelihood estimation.

<p align="center">
<img src="../figures/naive_bayes/learning_eq_3.gif">
</p>

<p align="center">
<img src="../figures/naive_bayes/learning_eq_4.gif">
</p>

### Inference

<p align="center">
<img src="../figures/naive_bayes/inference_eq_1.gif">
</p>

For numerical stability, you may instead compute

<p align="center">
<img src="../figures/naive_bayes/inference_eq_2.gif">
</p>

### Software

<p align="center">
<img src="../figures/naive_bayes/hyperparameters.png">
</p>

- **distribution**: the form of distribution for ![](../figures/naive_bayes/cond_dist.gif)
- **smoothing**: additive smoothing parameter
- **fit_prior**: whether to learn class prior probabilities from data
- **class_prior**: user-specified prior probabilities of the classes

Check out the documentation listed below to view the attributes that are available in sklearn but not exposed to the user in the software.

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/naive_bayes.html) on Naïve Bayes.
> 2. sklearn `LogisticRegression` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
