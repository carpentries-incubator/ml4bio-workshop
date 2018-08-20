# Support Vector Machine (SVM)

### Representation

We consider the scenario where training points belong to one of the two classes {-1, 1}.

**Linear SVM**

We are interested in finding a _linear_ hyperplane that separates positive points from negative points. The goal of linear SVM learning is to find a hyperplane that trades off between

- maximizing the **margin**
- minimizing the **slack variables**

where margin size measures how separated positive and negative points are,
and slack variables determine how far a point is from the boundary of the side where it should have lied (i.e. allow some imperfectness).

<p align="center">
<img src="../figures/svm/linear_svm_rep_fig.jpg">
</p>

**Kernel SVM**

Slack variables offer one option for classification of data that are not linearly separable. 
Another possibility is to map points into a higher dimensional space 
and find a hyperplane that separates the points in the new space.
Kernel SVM defines such mappings implicitly using **kernels**.
	 
<p align="center">
<img src="../figures/svm/kernel_svm_rep_fig.png">
</p>
	
<p align="center">
<img src="../figures/svm/kernels.jpg">
</p>	

### Learning

- **Linear SVM**

	Solve the constrained optimization problem
	
	<p align="center">
	<img src="../figures/svm/linear_obj.jpg">
	</p>
	
	Intuitively, the constant ![](../figures/svm/C.gif) accounts for the tradeoff between margin size and penalty on slack variables.
	
	We can rewrite the optimization problem above to obtain a more standard loss function.
	
	<p align="center">
	<img src="../figures/svm/loss_func.jpg">
	</p>
	

- **Kernel SVM**

	Similarly, construct the objective function as in the case of linear SVM 
	except using the data representation in the high-dimensional space.
	
	<p align="center">
	<img src="../figures/svm/kernel_obj_primal.jpg">
	</p>
	
Instead of solving the **primal** formulations above, we turn to their **dual** formulations, which are easier to solve because the constraints are linear and the kernel trick can be easily applied.
	
<p align="center">
<img src="../figures/svm/obj_dual.jpg">
</p>

where

<p align="center">
<img src="../figures/svm/linear_obj_dual_eq_4.gif">
</p>

for linear SVM, and

<p align="center">
<img src="../figures/svm/kernel_obj_dual_eq_4.gif">
</p>

for kernel SVM.

### Inference

- **Linear SVM**

	Since
	
	<p align="center">
	<img src="../figures/svm/linear_svm_inference_eq_1.gif">
	</p>

	we determine the class of a new point ![](../figures/svm/x.gif) by
	
	<p align="center">
	<img src="../figures/svm/linear_svm_inference_eq_2.gif">
	</p>
	
	where
	
	<p align="center">
	<img src="../figures/svm/sgn_func.gif">
	</p>

- **kernel SVM**

	<p align="center">
	<img src="../figures/svm/kernel_svm_inference_eq_1.gif">
	</p>
	

	Given a new point ![](../figures/svm/x.gif), we have
	
	<p align="center">
	<img src="../figures/svm/kernel_svm_inference_eq_2.gif">
	</p>
	
	and
	
	<p align="center">
	<img src="../figures/svm/kernel_svm_inference_eq_3.gif">
	</p>
	
	where ![](../figures/svm/x_star.gif) is some support vector.
	
	Therefore, we can determine the class of ![](../figures/svm/x.gif) by
	
	<p align="center">
	<img src="../figures/svm/kernel_svm_inference_eq_4.gif">
	</p>

### Software

<p align="center">
<img src="../figures/svm/hyperparameters.png">
</p>

- **penalty**: slack penalty strength ![](../figures/svm/C.gif) (i.e. larger values lead to fewer misclassifications.)
- **kernel**: the type of kernel used (_rbf_, _linear_, _poly_ or _sigmoid_)
- **degree**: the degree of polynomial in polynomial kernel
- **kernel_coef**: the cofficient ![](../figures/svm/gamma.gif) used in RBF, polynomial and sigmoid kernels
- **independent_term**: the independent term ![](../figures/svm/r.gif) in polynomial and sigmoid kernels
- **class_weight**: weights associated with the classes
	- _uniform_: every class receives the same weight.
	- _balanced_: class weights are inversely proportional to class frequencies.

Stopping criteria:

- **tol**: minimum reduction in loss required for optimization to continue.
- **max_iter**: maximum number of iterations allowed for the learning algorithm to converge. 

Check out the documentation listed below to view the attributes that are available in sklearn but not exposed to the user in the software.

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/svm.html) on SVM.
> 2. sklearn `SVC` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC).
