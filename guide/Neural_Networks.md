# Neural Networks

### Representation

<p align="center">
<img src="../figures/neural_network/binary_rep.jpg">
</p>

<p align="center">
<img src="../figures/neural_network/multiclass_rep.jpg">
</p>

### Inference

Given a new sample, we denote it by

<p align="center">
<img src="../figures/neural_network/inference_eq_0.gif">
</p>

where the first element is the **bias** term and the others are the feature values.

- **Binary problem**
	
	Consider a binary classification task with a positive class and a negative class.
	
	Denote the nodes in the hidden layer by ![](../figures/neural_network/a_i.gif) 
	and the incoming weights to ![](../figures/neural_network/a_i_2.gif) by
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_1.gif">
	</p>
	
	Then
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_2.gif">
	</p>
	
	and
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_3.gif">
	</p>
	
	where ![](../figures/neural_network/f.gif) is an activation function of your choice.
	
	Using similar notations, we have
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_4.gif">
	</p>
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_5.gif">
	</p>
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_6.gif">
	</p>
	
	and the probability that the new sample is positive is
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_7.gif">
	</p>

- **Multiclass problem**

	Consider a multiclass classification task with ![](../figures/neural_network/M.gif) classes ![](../figures/neural_network/classes.gif).
	
	Using the same notation as above, we have
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_1.gif">
	</p>
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_2.gif">
	</p>
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_3.gif">
	</p>
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_4.gif">
	</p>
	
	Then, define
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_8.gif">
	</p>
	
	We then get
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_9.gif">
	</p>
	
	and the probability that the new sample belongs to class ![](../figures/neural_network/i.gif) is
	
	<p align="center">
	<img src="../figures/neural_network/inference_eq_10.gif">
	</p>

- **Activation functions**

	<p align="center">
	<img src="../figures/neural_network/activation_fig.jpg">
	</p>

### Learning

- **Binary problem**

	<p align="center">
	<img src="../figures/neural_network/binary_loss.jpg">
	<img src="../figures/neural_network/cross_entropy.jpg">
	</p>

- **Multiclass problem**

	<p align="center">
	<img src="../figures/neural_network/multiclass_loss.jpg">
	</p>

In both cases, ![](../figures/neural_network/W.gif) is a vector containing all weights, 
and ![](../figures/neural_network/alpha.gif) is a constant 
that determines the strength of regularization.

### Example

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/neural_networks_supervised.html) on neural networks.
> 2. sklearn `MLPClassifier` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier).
> 3. Stanford CS231n [lecture note](http://cs231n.github.io/neural-networks-1/) on neural networks.


