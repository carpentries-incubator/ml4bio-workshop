# K-Nearest Neighbors

### Representation

<p align="center">
<img src="../fig/k_nearest_neighbors/rep_fig.jpg">
</p>

### Learning

None.

### Inference

Given a new point ![](../fig/k_nearest_neighbors/x.gif), find ![](../fig/k_nearest_neighbors/K.gif) training points 

<p align="center">
<img src="../fig/k_nearest_neighbors/nearest_pts.gif">
</p>

that are closest to it.

For continuous features, commonly used distance measures are

<p align="center">
<img src="../fig/k_nearest_neighbors/distances.jpg">
</p>

For discrete features, the similarity of two samples is measured by **Hamming distance**, 
which counts how many feature values are different between the two samples.

Then, the class of ![](../fig/k_nearest_neighbors/x.gif) is determined by

<p align="center">
<img src="../fig/k_nearest_neighbors/inference_eq.gif">
</p>

where

<p align="center">
<img src="../fig/k_nearest_neighbors/delta_func.gif">
</p>

In the weighted version, the neighbors are weighted by their distance to the new point.

<p align="center">
<img src="../fig/k_nearest_neighbors/weighted_inference_eq.gif">
</p>

### Software

<p align="center">
<img src="../fig/k_nearest_neighbors/hyperparameters.png">
</p>

- **n_neighbors**: the number of neighbors considered in a query
- **weights**: weights associated with the neighbors
	- _uniform_: every neighbor receives the same weight.
	- _distance_: weights are inversely proportional to the neighbors' distances (i.e. closer samples are more influential)
- **metric**: the distance metric used
	- _euclidean_: euclidean distance
	- _manhattan_: manhattan distance
	- _hamming_: hamming distance

Check out the documentation listed below to view the attributes that are available in sklearn but not exposed to the user in the software.


> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/neighbors.html) on Nearest Neighbors.
> 2. sklearn `KNeighborsClassifier` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
