# Software setup and a quick tour

> #### Questions
> 
> 1.   What do I need for running the software?
> 2.   What can I do with the software?
> 
> #### Objectives
> 
> 1.   Provide instructions on installation of the software.
> 2.   Explain how the software works through an example.

### Launch the software 


### A quick example

<img align="left" width="350" src="../screenshots/p1_before.png">

Here is what the left panel looks like before we start. In this stage, we will prepare the data for training and evaluating classifiers.

---

#### Step 1: load data

<img align="left" width="350" src="../screenshots/load_1.png">

Load a .csv file from the disc by clicking on <b>Select File</b>.
- A valid dataset consists of a number of feature columns and a single label column.
- The label column must be the last column.
- Each column has its corresponding name in its first row.

---

<img align="left" width="350" src="../screenshots/load_2.png">

After a valid file was loaded, the filename is shown next to the load button.
Now you may optionally load an unlabeled dataset that pairs up with the labeled one.
The two datasets must have the same features.
A summary of the dataset(s) is displayed. 
You may unfold the bullets in the summary to see more details.

---

#### Step 2: split data

<img align="left" width="350" src="../screenshots/train_test_split.png">

Decide how much data you want to set aside for testing.
You may adjust the percent of test data and decide whether or not to split the data in a stratified fashion.
We reserve 20% of the labeled dataset for testing.

---

#### Step 3: choose a validation method

<img align="left" width="350" src="../screenshots/validation.png">

Choose from one of the three supported validation strategies. 
We use 5-fold cross-validation with stratified sampling.

---

<img align="left" width="350" src="../screenshots/p1_after.png">

Now we are ready for training some classifiers. Click on <b>Next</b> to proceed to the next page.

---

<img align="left" width="350" src="../screenshots/p2_before.png">

In the second stage, we will train a number of classifiers 
and evaluate them based on the validation method you chose. 

---

#### Step 4: choose classifier type

<img align="left" width="350" src="../screenshots/classifier_type.png">

Use the drop-down menu to select a classifier type.
We select SVM (i.e. support vector machine).

---

#### Step 5: set classifier hyperparameters and train

<img align="left" width="350" src="../screenshots/svm_param.png">

A list of hyperparameters for an SVM shows up.
We first train an SVM using the default hyperparameter values.
You may give your classifier a name and add some comment at your own discretion.

---

<img align="left" width="350" src="../screenshots/before_train.png">

Now we are ready to train the classifier. 
If you want to start over, click on <b>Reset</b> and all values will be back to default. 
Otherwise, click on <b>Train</b> to proceed.



