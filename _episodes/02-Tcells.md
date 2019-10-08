---
title: "T-cells"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

### T cells 

Immunotherapy is a type of cancer treatment that uses the body’s own immune cells to boost natural defenses against cancer, and T cells are a popular target for immunotherapies. To fully optimize the effect of immunotherapy, T cell activation must be assessed at a single-cell level. Melissa Skala's group developed a label-free and non-destructive autofluorescence imaging method to easily and quickly acquire T cell intensity images. Therefore, an activated/quiescent (inactive) T cell classifier that uses autofluorescence intensity images can contribute to the applications of immunotherapy.

### Dataset description

This microscopy dataset includes gray scale images of two type of T cells: activated and quiescent (not activated). These T cells come from 6 donors.

|Activated|Quiescent|
|:---:|:---:|
|![CD3_2_act_2_6_66](https://user-images.githubusercontent.com/15007159/61666368-e4804d00-ac9c-11e9-9031-a3f9f6cfd7b1.png)|![CD8_2_noact_3_3_13](https://user-images.githubusercontent.com/15007159/61666346-d9c5b800-ac9c-11e9-9044-e13c218d0da0.png)|

Since WARF team is filing a patent for this study, we can only make these subsampled images public.

||Donor 1|Donor 2|Donor 3|Donor 4|Donor 5|Donor 6|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Full Activated|271|656|487|494|694|446|
|Full Quiescent|1631|152|1276|1629|252|589|
|Subsampled Activated|27|65|48|49|69|44|
|Subsampled Quiescent|163|15|127|162|25|58|

## Machine Learning Methods

The goal of this study is to develop a method to classify T cell activation stage (activated vs. quiescent). We have explored many classifiers.

| Model                                             | Description                                                                                                                                                                                      |
|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Frequency Classifier                              | Predict class probability using the class frequencies in the training set.                                                                                                                       |
| Logistic Regression with Pixel Intensity          | Regularized logistic regression model fitted with the image pixel intensity matrix (82 x 82). Regularization power λ of L1 penalty is tuned.                   |
| Logistic Regression with Total Intensity and Size | Regularized logistic regression model fitted with two numerical values: image total intensity and cell mask size. Regularization power λ of L1 penalty is tuned.                   |
| Logistic Regression with CellProfiler Features    | Regularized logistic regression model fitted with 123 features extracted from CellProfiler related to intensity, texture, and area. Regularization power λ of L1 penalty is tuned. |
| One-layer Fully Connected Neural Network          | Fully connected one-hidden-layer neural network with pixel intensity as input. Number of neurons, learning rate, and batch size are tuned.                                                       |
| LeNet CNN                                         | CNN with the LeNet architecture with pixel intensity as input. Learning rate and batch size are tuned.                                                                                           |
| Pre-trained CNN Off-the-shelf Model               | Freeze layers of a pre-trained Inception v3 CNN. Train a final added layer from scratch with extracted off-the-shelf features. Learning rate and batch size are tuned.                           |
| Pre-trained CNN with Fine-tuning                  | Fine-tune the last n layers of a pre-trained Inception v3 CNN. The layer number n, learning rate, and batch size are tuned.                                                                  |

Here is the test result of these classifiers:

![metric](https://user-images.githubusercontent.com/15007159/61667809-9ff6b080-aca0-11e9-9f11-875f49b94344.png)
