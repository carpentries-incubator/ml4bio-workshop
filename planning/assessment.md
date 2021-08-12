## Assessment

Instructions: Please read these adapted excerpts from the paper, "RFMirTarget: predicting human microRNA target genes with a random forest classifier”, fill in the chart on page 2, and answer the questions on page 3. Please note that the excerpts have been adapted for this activity, including significantly altering the experimental procedure and possibly introducing errors. Formulas for performance metrics can be found in lesson 03

> In this paper we discuss and explore the predictive power of RFMirTarget, a ML approach for predicting human miRNAs target genes based on the random forests algorithm. 

> RFMirTarget is trained with a set of positive and negative examples of miRNA-target pairs that is pre-processed by the software miRanda in order to identify the actual interacting sites between each miRNA-mRNA pair and prepare the data set for feature extraction. The alignments provided by miRanda are the source for extracting features, which in turn are used to train the random forest classifier.

> We train RFMirTarget with experimentally verified examples of human miRNA-target pairs. The data set is composed of 289 biologically validated positive examples extracted from miRecords database and 289 systematically identified tissue-specific negative examples. These examples were split into a training set consisting of 80% of instances and a testing set consisting of 20% of the data. Training and testing sets were balanced to contain equal proportions of positive and negative examples. 

> The set of descriptive features used to train RFMirTarget is divided into three categories: alignment features, thermodynamics features, and structural features. 
>      1.      Alignment features: Score and length of the miRNA-target alignment as evaluated by miRanda. 
>      2.      Thermodynamics features: Evaluation of the minimum free energy (MFE) of the complete miRNA-target alignment computed by RNAduplex. 
>      3.      Position-based features: Evaluation of each base pair from the 5'-most position of the miRNA up to the 30th position of the alignment, assigning nominal values to designate the kind of base pairing in each position: a G:C match, an A:U match, a G:U wobble pair, a gap and a mismatch.

> To train this RF model we adopt the standard number of trees suggested by the randomForest R package, namely 500 trees. In order to determine the number of features each tree in the random forest should have we experimented giving trees between 1 and 35 features. We used 12 features for the final model as it had the highest specificity on the testing set.

> The performance of RFMirTarget is assessed by computing the total prediction accuracy (ACC), specificity (SPE), sensitivity (SEN). Training set performance was ACC: 92.21, SEN: 93.73, SPE: 91.11. The classification results drawn from our experimental procedure was ACC: 78.98, SEN: 80.04, and SPE: 77.48.

> *Adapted from: Mendoza, Mariana R., et al. "RFMirTarget: predicting human microRNA target genes with a random forest classifier." PloS one 8.7 (2013): e70153.*

1. In this excerpt, what is...
 - ...the class label?
 - ...the number of instances?
 - ...the model?
 - ...the evaluation metric(s) used?

2. How was the data split between training, testing, and validation?

3. Is there any evidence of data leakage?

4. Is there any evidence of overfitting?

5. Was the evaluation metric(s) used appropriate?

6. How did the model perform?

7. Do you trust the validity of these results?
