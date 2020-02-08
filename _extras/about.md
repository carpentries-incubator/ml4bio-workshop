---
title: About ml4bio software
---
{% include links.md %}

### Software environment details
 
Anaconda includes software that enables you to run Python programs as well as additional tools for managing software environments, programming in Python, and integrating code with textual descriptions and results in Jupyter notebooks.
The software environments are managed by conda, one of the tools included with Anaconda.
An environment is a collection of specific versions of Python packages.
These are all stored in a directory that conda manages.
Having multiple environments allows you to use different versions of the same package for different projects.

The ml4bio install scripts create a new conda environment.
This environment, which is named `ml4bio`, contains the latest version of the `ml4bio` Python package as well as suitable versions of other Python packages that it requires.
The `ml4bio` code may be incompatible with older or newer versions of the Python packages it uses.
The environment makes it easy for you use a collection of Python packages that work together.

The most important required Python package that ml4bio uses is called [scikit-learn](http://scikit-learn.org/).
This is a popular general purpose machine learning package.
When you use the ml4bio graphical interface, it calls functions in scikit-learn to train classifiers and make predictions.

### Advanced users
Advanced users who already have Python installed can install the [required packages](https://github.com/gitter-lab/ml-bio-workshop/blob/master/scripts/README.md) through pip.
Then launch ml4bio from the command line with the command `ml4bio`.

### A quick tour

Once you open the ml4bio software, you will see the following interface.
It may look slightly different in newer versions of ml4bio.

<p align="center">
<img src="{{ page.root }}/fig/software/window.png">
</p>

The interface can be divided into three areas:

- **Red area.**
This is the main area you operate on.
It consists of three pages.
The first page allows you to load data, split data and choose a validation method (steps 1,2 and 3).
The second page is where you train classifiers (steps 4 and 5).
The last page is for testing and prediction (steps 6 and 7).

- **Yellow area.**
This is where trained classifiers are listed.
An entry will be added after a classifier is trained.
Each entry contains six useful performance metrics
that assist you in classifier evaluation and selection.
You will find this area helpful when you train, evaluate and test classifiers.

- **Blue area.**
This area is for data provenance and visualization.
Once a classifier in the yellow area is selected,
you may examine its hyperparameters, performance metrics and plots
using the interface provided in this area.

We illustrate the use of the software by working through an example.
The interface may differ slightly on different operating systems.

#### Step 1: Load data

Load a .csv file from the disc by clicking on **Select File...**.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/before_load.png">
</p>

<p align="center">
<img width="800" src="{{ page.root }}/fig/software/file_1.png">
</p>

Note that

- A valid dataset consists of a number of feature columns and a single label column.
- The label column must be the last column.
- A header is required.
- The delimiter can be ',' or an empty space.

<p align="center">
<img width="700" src="{{ page.root }}/fig/software/format.png">
</p>

After a valid labeled dataset is loaded, the file name will be shown next to **Select File...**.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/after_load.png">
</p>

You may optionally load an unlabeled dataset that pairs up with the labeled one.
The two datasets _must_ have the same column names
except that the label column is missing in the unlabeled data.
After you trained a classifier, you may make predictions on the unlabeled data.

A summary of the dataset(s) is displayed in a tree structure.
You may unfold the bullets to see more details.
There are a total of 100 samples in the dataset we loaded.
Half of them belong to class 0, and the other half belongs to class 1.
The dataset has only two features, namely x and y.
Both of them are continous.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/data_summary.png">
</p>

#### Step 2: Split data

Decide how much data you want to set aside for testing.
You may adjust the percent of test data
and decide whether or not to split the data in a stratified fashion.
We reserve 20% of the labeled dataset for testing by default.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/train_test_split.png">
</p>

#### Step 3: Choose a validation method

Choose from one of the three supported validation strategies.
Adjust the values accordingly.
We use 5-fold cross-validation with stratified sampling by default.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/validation.png">
</p>

Here is what the left panel (i.e. the red area) looks like after step 3.
We are ready for training classifiers.
Click on **Next** to go to the next page.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/p1_after.png">
</p>

#### Step 4: Set up a classifier

Use the drop-down menu to select a classifier type.
We select SVM (i.e. support vector machine).

<p align="center"><img width="350" src="{{ page.root }}/fig/software/classifier_type.png"></p>

A list of hyperparameters for an SVM shows up.
The meaning of the hyperparameters will be introduced in a later section.
We first train an SVM using the default hyperparameters.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/svm_param.png">
</p>

You may give your classifier a name and add a comment.
If you do not specify a name, the software will use "classifier\_[int]" as its default name.
For example, if the classifier is the third one you trained, its default name is "classifier\_3".

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/name_comment.png">
</p>

#### Step 5: Train and evaluate classifiers

Now everything has been set up for training an SVM.
If you changed the hyperparameters but want to start over,
click on **Reset**.
The hyperparameters will be back to default.
Otherwise, click on **Train**.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/before_train.png">
</p>

In the yellow area, a new entry was added to the list.
It includes the name, type and the six performance metrics of the newly trained classifier.
_Note that the metrics are all with respect to the type of data, training or validation,
shown at the top-right corner of the yellow area._

<p align="center">
<img src="{{ page.root }}/fig/software/after_train.png">
</p>

There are a few things you can do at this stage:

- **Examine classifier summary.**
A summary of the newly trained classifier is presented in the blue area.
As with data summary, you may unfold the bullets to see more details.

<p align="center">
<img src="{{ page.root }}/fig/software/param_expand.png">
<img src="{{ page.root }}/fig/software/perform_expand.png">
</p>

The left figure shows a complete list of classifier hyperparameters.
Some of them were exposed to you for tuning, others were fixed by the software.
You may learn more about the hyperparameters from the [sklearn documentation](http://scikit-learn.org/stable/documentation.html).

The right figure shows the classifier's performance with respect to different types of data.
For each metric, classwise values as well as their average are computed.
For example, the precision of the classifier on class 0 on the training data is 0.89.
The classifier's overall precision on the training data is 0.9.

- **Examine plots.**
Three types of plots that reflect the classifier's performance are always available.
The data plot is only available when the dataset contains exactly two continuous features.
_Note that the plots are all with respect to the type of data
shown at the top-right corner of the yellow area._

<p align="center">
<img width='200' src="{{ page.root }}/fig/software/data_plot.png">
<img width='200' src="{{ page.root }}/fig/software/confusion_matrix.png">
</p>

Shown on the left is a scatter plot of the training data and contours of the decision regions.
The darker the color, the more confident the classifier is.
Shown on the right is the confusion matrix.

<p align="center">
<img width='200' src="{{ page.root }}/fig/software/roc.png">
<img width='200' src="{{ page.root }}/fig/software/prc.png">
</p>

The left figure includes ROC curves and the right one includes precision-recall curves.
A curve is plotted for each class.
The average curve is the unweighted average of all classwise curves.

- **Switch between performance on different data types.**
You may want to compare the classifier's performance on training and validation data
to see how well it generalizes.
Use the drop-down menu at the top-right corner of the yellow area
to switch between performance on different types of data.

<p align="center">
<img src="{{ page.root }}/fig/software/switch_metric.png">
</p>

Typically, you will train and evaluate many classifiers
before you find one that you are satisfied with.
Using the software, you may train as many classifiers as you want.
You may click on the header of the list to sort the classifiers by a particular metric.
For example, if you click on **Accuracy**,
the classifiers will be listed in descending order of accuracy.

<p align="center">
<img src="{{ page.root }}/fig/software/after_sort.png">
</p>

It seems that the SVM we trained first is the second best classifier in terms of accuracy.
The best one is a k-nearest neighbor classifier that achieves 91% accuracy on the validation data.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/p2_after.png">
</p>

Let's say we are happy about the k-nearest neighbor classifier.
Click on **Next** to proceed to the next page.
However, if you want to change, say, the validation method,
you may click on **Back**, which will bring you to the previous page.
_Be careful, because all trained classfiers will be lost if you do so._

#### Step 6: Test a classifier

We are done with training and ready to select a classifier for testing.
However, if you changed your mind and decided to train more classifiers,
you have a chance to return to the previous page by clicking on **Back**.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/p3_before.png">
</p>

To select a classifier, you may let the software pick one for you by specifying a metric.
In this case, the software will select the best classifier with respect to that metric.
Otherwise, you may pick a classifier on your own.
We let the software select the classifier with the highest accuracy.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/before_test.png">
</p>

After a classifier is selected, its name will show up.
Double-check that it is the one you want to test.
Now the **Test** button is enabled, and you may click on it to test the selected classifier.
_Note that once you hit **Test**,
you are no longer allowed to go back and train more classifiers._

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/test.png">
</p>

Now the only classifier in the list is the tested one.
Note that the software is showing the classifier's performance on the test data.
You may examine the performance using either the summary or the plots.

<p align="center">
<img src="{{ page.root }}/fig/software/after_test.png">
</p>

Although not recommended, you are allowed to test any of the other trained classifiers.
Switch the type of data to either training or validation data
and you will see the complete list of trained classifiers.
Select a classifier and test it.

<p align="center">
<img src="{{ page.root }}/fig/software/test_more.png">
</p>

#### Step 7: Make predictions

Optionally, you may make predictions on the unlabeled data you uploaded and save the results.
If no unlabeled data exists, the **Predict** button is disabled.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/predict.png">
</p>

Finally, finish your work by clicking on **Finish**.
A message box will show up and you may choose from closing the software
or modeling a different dataset.

<p align="center">
<img width="350" src="{{ page.root }}/fig/software/finish.png">
</p>
