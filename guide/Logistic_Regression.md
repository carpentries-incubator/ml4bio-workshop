# Logistic Regression

### Representation

<p align="center">
<img src="../figures/logistic_regression/binary_rep.jpg">
</p>

<p align="center">
<img src="../figures/logistic_regression/multiclass_rep.jpg">
</p>

### Inference

Given a new sample, we denote it by

<p align="center">
<img src="../figures/logistic_regression/inference_eq_0.gif">
</p>

where the first element is the **bias** term and the others are the feature values.

- **Binary problem**
	
	Consider a binary classification task with a positive class and a negative class.
	
	Denote

	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_1.gif">
	</p>
	
	Then
	
	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_2.gif">
	</p>
	
	and the probability that the new sample is positive is
	
	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_3.gif">
	</p>

- **Multiclass problem**
	
	Consider a multiclass classification task with classes ![](../figures/logistic_regression/classes.gif).
	
	Denote
	
	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_4.gif">
	</p>
	
	Then
	
	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_5.gif">
	</p>
	
	and the probability that the new sample belongs to class ![](../figures/logistic_regression/i.gif) is
	
	<p align="center">
	<img src="../figures/logistic_regression/inference_eq_6.gif">
	</p>

### Learning

- **Binary problem**

	<p align="center">
	<img src="../figures/logistic_regression/binary_loss.jpg">

	<img src="../figures/logistic_regression/cross_entropy.jpg">
	</p>

- **Multiclass problem**

	<p align="center">
	<img src="../figures/logistic_regression/multiclass_loss.jpg">
	</p>

In both cases, ![](../figures/logistic_regression/W.gif) is a vector containing all weights, 
and ![](../figures/logistic_regression/alpha.gif) is a constant 
that determines the strength of regularization.

### Example

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/linear_model.html) on linear models (including Logistic Regression).
> 2. sklearn `LogisticRegression` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


