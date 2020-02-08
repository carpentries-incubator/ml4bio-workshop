# gh-pages branch README

# Machine Learning for Biology

This repository contains materials for the Machine Learning for Biology workshop.
It generates the corresponding lesson website from [The Carpentries](https://carpentries.org/) repertoire of lessons. 

## Contributing

We welcome contributions to improve the workshop!
The best way to provide suggestions or propose contributions is in the [GitHub issues](https://github.com/gitter-lab/ml-bio-workshop/issues).

## Maintainers

Current maintainers of this lesson are 

* [@cmilica](https://github.com/cmilica)
* [@csmagnano](https://github.com/csmagnano)
* [@agitter](https://github.com/agitter)

## Authors

A list of contributors to the lesson can be found in [AUTHORS](AUTHORS) and the [GitHub contributors](https://github.com/gitter-lab/ml-bio-workshop/graphs/contributors).

## Citation

To cite this lesson, please consult with [CITATION](CITATION)

[lesson-example]: https://carpentries.github.io/lesson-example





# master branch README

# Machine Learning (ML) for Biologists Workshop
[![Build Status](https://travis-ci.org/gitter-lab/ml-bio-workshop.svg?branch=master)](https://travis-ci.org/gitter-lab/ml-bio-workshop)
[![Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gitter-lab/ml-bio-workshop/master?filepath=illustration.ipynb)

Recent advances in high-throughput technologies have led to rapid growth in the amount of omics data.
These massive datasets, accompanied by numerous data points enriched over the past decades, are extremely valuable in the discovery of complex biological structures, but also present big challenges in various aspects of data analysis.
This workshop introduces biologists to machine learning, a powerful set of models for drawing inference from big data.
Upon completion of the workshop, attendees will gain expertise in building simple predictive models and become more comfortable to collaborate with computational experts on more complicated data analysis tasks.

## Getting started
Visit the [software setup guide](guide/setup.md) for instructions on installing the ML4Bio software and downloading the example datasets, guides, and slides contained in this repository.

## Overview
This repository contains the following files and subdirectories:
- [`binder`](binder): a configuration file to enable running the `illustration.ipynb` notebook interactively using [Binder](https://mybinder.org/)
- [`data`](data): example datasets to use with the ML4Bio classification software, including both real and toy (illustrative simulated) datasets
- [`figures`](figures): figures for the guides
- [`guide`](guide): tutorials about the ML4Bio software, machine learning concepts, and various classifiers
- [`scripts`](scripts): ML4Bio installation scripts
- [`illustration.ipynb`](illustration.ipynb): a Jupyter notebook the demonstrates the machine learned workshop in Python code
- `workshop_slides.pptx`: slides for the machine learning for biology workshop

The [ml4bio repository](https://github.com/gitter-lab/ml4bio) contains the Python software.

## License

The workshop guides created in this repository are licensed under the Creative Commons Attribution 4.0 International Public License.
However, the guides also contain [images from third-party sources](figures/third_party_figures), as noted in the image links and guide references.
See the linked original image sources for their licenses and terms of reuse.

The work-in-progress website on the `gh-pages` branch is derived from [The Carpentries lesson template](https://github.com/carpentries/styles).
The template is Copyright © 2018–2019 [The Carpentries](https://carpentries.org/) and [licensed CC BY 4.0](https://github.com/carpentries/styles/blob/gh-pages/LICENSE.md).
