# Random Forest

### Representation

<p align="center">
<img src="../figures/random_forest/rep_fig.jpg">
</p>

### Learning

Given the full training data ![](../figures/random_forest/D.gif) 
and the feature set ![](../figures/random_forest/F.gif), we train ![](../figures/random_forest/T.gif) randomized decision tree.

For tree ![](../figures/random_forest/i.gif),

- Sample ![](../figures/random_forest/D.gif) with replacement to create a new dataset ![](../figures/random_forest/D_i.gif).

- To choose a split at a node, 
sample ![](../figures/random_forest/m.gif) features without replacement from ![](../figures/random_forest/F.gif) (where ![](../figures/random_forest/m_f.gif)) to create a subset of features ![](../figures/random_forest/F_m.gif). Choose the best split from ![](../figures/random_forest/F_m.gif).

The intuition is that each randomized tree has increasing _bias_ compared to a tree that utilizes full data and feature set, but the **ensemble** of randomized trees enables reduction in _variance_ in performance. 

As with decision tree learning, we can regularize the randomized trees by limiting their maximum depth, the minimum number of samples at a leaf, the total number of leaves, etc.

### Inference

Given a new sample ![](../figures/random_forest/x.gif), feed it to the trees in the forest.
Each tree will produce a class label for ![](../figures/random_forest/x.gif), and the final label is determined by a majority vote. Sometimes (as in sklearn), each tree will produce a probability for each class instead of a concrete label. In such case, we average over the predicted probabilities for each class across all trees. The final label is the class with highest average probability.

### Example

> #### Further readings
> 1. sklearn [tutorial](http://scikit-learn.org/stable/modules/ensemble.html) on Ensemble methods (including Random Forest).
> 2. sklearn `RandomForestClassifier` [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
