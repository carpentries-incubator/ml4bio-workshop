## T Cell Image Data

This directory contains extracted image features used in the Gitter Lab's [T cell classification](https://github.com/gitter-lab/t-cell-classification) 
project. Each CSV file corresponds to one version of features used in the study.
Features are extracted from 6 donors of [10% subsampled images](https://github.com/gitter-lab/t-cell-classification/tree/master/images).

|File|Description|
|:---:|:---:|
|`size_intensity_feature.csv`|This file includes two features extracted from subsample images: cell mask size and cell image total intensity.|
|`cellprofiler_feature.csv`|This file includes 123 features extracted from subsampled images using the software CellProfiler. These 123 features covers pipelines `MeaureObjectSizeShape`, `MeasureObjectIntensity`, and `MeasureTexture`.|

## Notes

- In this project, researchers are interested in classifying T cells' two
activation states: activated (positive) and quiescent (negative).
- There are image feature differences across 6 donors, therefore the donor ID is
included as the first column of each CSV file.
- The row indices of `size_intensity_feature.csv` and `cellprofiler_feature.csv`
are the same. For example, row 100 in both tables links to the same image.
- The versions of the files with `_donors` contain the same features and labels
but also include a donor identifier feature.

## Citation

Manuscript "Classifying T cell activity in autofluorescence intensity images with convolutional neural networks" by Zijie J. Wang, Alex J. Walsh, Melissa C. Skala, and Anthony Gitter coming soon.
