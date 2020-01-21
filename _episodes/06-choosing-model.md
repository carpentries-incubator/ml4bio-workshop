---
title: "How to Choose a Model"
teaching: 10
exercises: 5
questions:
- "How do you evaluate the performance of a model?"
objectives:
keypoints:
- The choice of evaluation metric depends on what the data's class balance is, and what we want the model to succeed at.
- Comparing performance on the validation set with the right metric is an effective way to select a classifier and hyperparameter settings.
mathjax: true
---

### Model Selection

Choosing the proper machine learning model for a given task requires knowledge of both machine learning models and the domain of the task. 
Finding _the best_ model for a new task in machine learning is often a research question in itself.
Finding a model that performs _reasonably well_, however, can often be accomplished by carefully considering the task domain and a little trial and error with the validation set. 

Some of the questions to consider when choosing a model are:

* How much data is there to train with?
* Does the data contain about the same number of each class? 
* How many features does the dataset have? Are all of the features relevant, or might some of them not be related to the data's class?
* What types are the features (numeric, categorical, image, text)?
* What might the decision boundary look like? Is the data likely linearly separable?
* How noisy is the data? 

### Evaluation Metrics

Arguably the most important part of choosing a model is evaluating it to see how well it performs. So far we've been looking at metrics such as accuracy, but let's take a look at how we think about metrics in machine learning.

In the binary classification setting (where there are only two classes we're trying to predict, such as activated or quiescent T-cells) we can group all possible predictions our classifier makes into four categories. This is organized into a table called the _confusion matrix_:

_Comment: We'll probably want to replace these with images in the same style as the rest of the workshop._

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/confusionMatrix_plain.png">
</p>


Here, all possible types of predictions are split by 1) What the actual, true class is and 2) what the predicted class is, what our classifier thinks the truth is. This results in the 4 entries of the confusion matrix, two of which means our classifier got something right:

- True Positives (TP): These instances are actually true (activated) and have been correctly predicted to be true.
- True Negatives (TN): These instances are actually false (quiescent) and have been correctly predicted to be false. 

And two of which means our classifier got something wrong:

- False Positives (FP): These are instances which are actually false but our classifier predicted to be true. False positives are sometimes called type I errors or $\alpha$ errors. 
- False Negatives (FN): These are instances which are actually true but our classifier predicted to be false. False negatives are sometimes called type II errors or $\beta$ errors. 

Almost all evaluation metrics used in machine learning can be derived from the entries in the truth table, typically as a ratio of two sets of entries. For instance, accuracy is defined as the percent of instances the classifier got right.

So to calculate accuracy we take the number of things we got right, which is the number of true positives and the number of true negatives: 

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/confusionMatrix_P.png">
</p>


And divide it by the number total entries in the table, which is all four entries:

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/confusionMatrix_All.png">
</p>

Thus, accuracy is defined as $$\frac{TP + TN}{TP+FP+TN+FN}$$
_Comment: We need to debug equations. Testing with this one first._

We can see accuracy as estimating the answer to the question _How likely is our classifier to get a single instance right?_ However, for many models this might not be the right question. 

An example of a different question we might want to ask about a model would be _If our classifier predicts something to be true, how likely is it to be right?_

To answer this question we would look at everything we predicted to be true, which is the true positive and false positives:

<p align="center">
<img width="450" src="https://raw.githubusercontent.com/gitter-lab/ml-bio-workshop/gh-pages/assets/confusionMatrix_tpfp.png">
</p>


We would then calculate the percent of these predictions that were correct, which are the true positives. Thus, to answer this question we would use the metric $\frac{TP}{TP + FP}$

This metric is called _precision_ in machine learning (and may be different from the definition of precision you use in a laboratory setting).

> ## Scenario
>
> You are designing a machine learning system for discovering existing drugs which may be effective in treating Malaria, focusing on the parasite _Plasmodium falciparum_. Your system takes in information on an FDA approved drug's chemical structure, and predicts whether or not a drug may interact with _P. falciparum_. The machine learning system will be used as an initial screening step; drugs classified as interacting by your system will be flagged and tested in the lab. The vast majority of drugs will not be useful in treating _P. falciparum_.
> With your neighbor, talk through the following questions:
> 1. Which entries in the confusion matrix are most important in this machine learning task? Is one type of correct or one type of incorrect prediction more important than the other for your machine learning system?
> 2. Imagine a classifier that predicted that all drugs are non-interacting. If you evaluated this classifier using the entire catalog of FDA approved drugs, what would the accuracy look like?
> 3. What metric or couple of metrics would you use to evaluate your machine learning system?
>
>
> Load the `simulated-drug-discovery` dataset from the `data` folder into the ML4Bio software. Trying training a logistic regression classifier on the dataset. Which metrics seem to accurately reflect the performance of the classifier?
{: .callout}

Common Metrics:

| Name                                   	|    Formula    	                |
|----------------------------------------	|:--------------------------------:	|
| Accuracy                               	| $\frac{TP + TN}{TP+FP+TN+FN}$ 	|
| Precision (Positive Predictive Value) 	| $\frac{TP}{TP + FP}$         	|
| Recall (Sensitivity)                   	| $\frac{TP}{TP + FN}$ 	        |
| True Negative Rate (Specificity)       	| $\frac{TN}{TN + FP}$          	|
| False Positive Rate                    	| $\frac{FP}{FP + TN}$         	|
| F1 Score                               	| $\frac{2TP}{TP + TN + FP + FN}$ |
