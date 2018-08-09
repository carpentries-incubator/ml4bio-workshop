# K-Nearest Neighbors

###Representation

<p align="center">
<img src="../figures/k_nearest_neighbors/rep_fig.jpg">
</p>

###Learning

None.

###Inference

Given a new point ![](../figures/k_nearest_neighbors/x.gif), find ![](../figures/k_nearest_neighbors/K.gif) training points 

<p align="center">
<img src="../figures/k_nearest_neighbors/nearest_pts.gif">
</p>

that are closest to it.

For continuous features, commonly used distance measures are

<p align="center">
<img src="../figures/k_nearest_neighbors/distances.jpg">
</p>

For discrete features, the similarity of two samples is measured by **Hamming distance**, 
which counts how many feature values are different between the two samples.

Then, the class of ![](../figures/k_nearest_neighbors/x.gif) is determined by

<p align="center">
<img src="../figures/k_nearest_neighbors/inference_eq.gif">
</p>

where

<p align="center">
<img src="../figures/k_nearest_neighbors/delta_func.gif">
</p>

In the weighted version, the neighbors are weighted by their distance to the new point.

<p align="center">
<img src="../figures/k_nearest_neighbors/weighted_inference_eq.gif">
</p>

###Example

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/neighbors.html) on Nearest Neighbors.
> 2. sklearn `KNeighborsClassifier` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
