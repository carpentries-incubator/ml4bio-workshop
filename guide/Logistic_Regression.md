# Logistic Regression

### Representation

<p align="center">
<img src="../fig/logistic_regression/binary_rep.jpg">
</p>

<p align="center">
<img src="../fig/logistic_regression/multiclass_rep.jpg">
</p>

### Inference

Given a new sample, we denote it by

<p align="center">
<img src="../fig/logistic_regression/inference_eq_0.gif">
</p>

where the first element is the **bias** term and the others are the feature values.

- **Binary problem**
	
	Consider a binary classification task with a positive class and a negative class.
	
	Denote

	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_1.gif">
	</p>
	
	Then
	
	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_2.gif">
	</p>
	
	and the probability that the new sample is positive is
	
	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_3.gif">
	</p>

- **Multiclass problem**
	
	Consider a multiclass classification task with classes ![](../fig/logistic_regression/classes.gif).
	
	Denote
	
	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_4.gif">
	</p>
	
	Then
	
	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_5.gif">
	</p>
	
	and the probability that the new sample belongs to class ![](../fig/logistic_regression/i.gif) is
	
	<p align="center">
	<img src="../fig/logistic_regression/inference_eq_6.gif">
	</p>

### Learning

- **Binary problem**

	<p align="center">
	<img src="../fig/logistic_regression/binary_loss.jpg">

	<img src="../fig/logistic_regression/cross_entropy.jpg">
	</p>

- **Multiclass problem**

	<p align="center">
	<img src="../fig/logistic_regression/multiclass_loss.jpg">
	</p>

In both cases, ![](../fig/logistic_regression/W.gif) is a vector containing all weights, 
and ![](../fig/logistic_regression/alpha.gif) is a constant 
that determines the strength of regularization.

### Software

<p align="center">
<img src="../fig/logistic_regression/hyperparameters.png">
</p>

- **penalty_type**: the norm used in the regularization term (_L1_ or _L2_)
- **penalty**: inverse of regularization strength ![](../fig/logistic_regression/alpha.gif) (i.e. larger values lead to weaker regularization.)
- **fit_intercept**: whether to use a bias term
- **intercept_scaling**: scale of the bias term
- **solver**: learning algorithm used to optimize the loss function
- **multi_class**: mode for multiclass problems
	- _ovr_: one vs. all (one classifier for each class)
	- _multinomial_: one classifier for all classes
- **class_weight**: weights associated with the classes
	- _uniform_: every class receives the same weight.
	- _balanced_: class weights are inversely proportional to class frequencies.

Stopping criteria:

- **tol**: minimum reduction in loss required for optimization to continue.
- **max_iter**: maximum number of iterations allowed for the learning algorithm to converge. 

Check out the documentation listed below to view the attributes that are available in sklearn but not exposed to the user in the software.

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/linear_model.html) on linear models (including Logistic Regression).
> 2. sklearn `LogisticRegression` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


