# Machine Learning (ML) for Biologists Workshop
[![Build Status](https://travis-ci.org/gitter-lab/ml-bio-workshop.svg?branch=master)](https://travis-ci.org/gitter-lab/ml-bio-workshop)

Recent advances in high-throughput technologies have led to rapid growth in the amount of omics data.
These massive datasets, accompanied by numerous data points enriched over the past decades, are extremely valuable in the discovery of complex biological structures, but also present big challenges in various aspects of data analysis.
This workshop introduces biologists to machine learning, a powerful set of models for drawing inference from big data.
Upon completion of the workshop, attendees will gain expertise in building simple predictive models and become more comfortable to collaborate with computational experts on more complicated data analysis tasks.

## Overview
This repository contains the following files and subdirectories:
- [`data`](data): example datasets to use with the ML4Bio classification software, including both real and toy (illustrative simulated) datasets
- [`docs`](docs): documentation for the ML4Bio source code, primarily for developers
- [`figures`](figures): figures for the guides
- [`guide`](guide): tutorials about the ML4Bio software, machine learning concepts, and various classifiers
- [`src`](src): the ML4Bio software
- [`illustration.ipynb`](illustration.ipynb): a Jupyter notebook the demonstrates the machine learned workshop in Python code
- `workshop_slides.pptx`: slides for the machine learning for biology workshop

The [ml4bio repository](https://github.com/gitter-lab/ml4bio) contains the Python software.

## Guideline

### Fundamentals of ML

* Task definition: classification (main focus), regression, clustering, etc.
* Data representation:
  * Features: categorical, numerical (discrete/continuous)
  * Labels: univariate (binary/multi-class), multivariate
  * File formats: txt, json, csv, etc.
  * Input (feature vectors) -> model (functions) -> output (predictions)
  * Learning: parameter estimation
  * Prediction: function evaluation

### ML in Biological Data Analysis: Pipeline

* Problem formulation
* Data preparation: cleaning, integration, etc.
* Iteratively:
  * Feature engineering
  * Model building: training, validation, evaluation
  * Model selection
  * Feature selection
* Testing
* Interpretation

### Models for Classification: An Introduction

* A GUI (developed using PyQt5) with two modes:
  * Playground: built-in toy datasets, data/model visualization
    (reference: http://playground.tensorflow.org)
  * Production: user-supplied datasets, no visualization
    (reference: https://github.com/gmiaslab/ClassificaIO)
* Models to be included:
  * Decision trees (viewed as a special case of RF)
  * Random forests
  * K-nearest neighbors
  * Logistic regression (viewed as a special case of NN)
  * Neural networks
  * Support vector machines
  * Naive bayes

### Models for Classification: Training and Evaluation

Expand on the following topics:
* Training set, validation set, test set
* Stopping criteria
* K-fold cross-validation
* Leave-one-out
* Evaluation metrics: accuracy (error rate), precision, recall, F1, etc.
* Graphics: ROC, Precision-recall curve, confusion matrix

Unique issues in biological application:
* Small size of data
* How to obtain negative examples?
* Unbalanced data
* Sparsity
* Cost-sensitive analysis (How to pick threshold?)

### Model Selection

Expand on the following topics:
* Overfitting
* Outliers

### Feature selection

### Model interpretation

### Feature engineering


to be continued...

## License

The workshop guides created in this repository are licensed under the Creative Commons Attribution 4.0 International Public License.
However, the guides also contain [images from third-party sources](figures/third_party_figures), as noted in the image links and guide references.
See the linked original image sources for their licenses and terms of reuse.
