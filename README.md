# Illuminating the role of cold-pools in structuring shallow convection
![Static Badge](https://img.shields.io/badge/Studio-Link_to_NN_training_metrics?logo=dvc&color=white&link=https%3A%2F%2Fstudio.iterative.ai%2Fuser%2FobservingClouds%2Fprojects%2Fcold_pool_detection-yg8z322abh)

This repository is work in progress and developed during the UW eScience Data Science Incubator project 2024.

## Project description

![project_teaser_img](https://github.com/observingClouds/cold_pool_detection/assets/43613877/243cff29-9738-4424-a1b9-ecc841941847)

Shallow convection, like the stratocumulus decks off the Washington coast, is responsible for a large
portion of the uncertainty in climate projections, thus a better understanding of
their processes is crucial. Advances in computational resources allow for ever increasing resolutions of
climate simulations, yet the resolutions remain too coarse to simulate these clouds and their underlying
processes explicitly. Parameterizations – simple algorithms or empiric relationships that estimate the
unresolved processes based on the resolved processes – need to be refined with the increase in resolution
as they no longer hold true. To develop these new parameterizations, the formation processes of these
shallow clouds need to be understood at finer and finer detail. A detail that is currently left out in these
parameterizations is the fact that these clouds can occur in a variety of spatial patterns. To simulate
these clouds and their cooling effect correctly in the current and future climate, these patterns are crucial
to represent correctly.

In order to develop a better parameterization of these clouds in our climate models, we need to improve
our understanding on how these different patterns of cloudiness form. Previous studies suggest that
precipitation drastically influences these patterns, in particular through the generation of so-called cold
pools. These cold pools (marked in red in the satellite image) that are areas of cold air and form due to the evaporation
of precipitation are able to redistribute clouds by suppressing them within the cold pool and generating
new convection at their edges. The identification of these cold pools in satellite observations will
provide valuable information to better understand the formation of different cloud patterns and
ultimately lead to an improved parameterization of shallow convection.

Here we utilize several data sources to generate ground-truth cold-pool labels and train a neural network that is capable to identify individual cold pools in satellite imagery.


## Reproducability / Usage
This repository uses [dvc](dvc.org) to keep track of workflows and data.

```bash
# Reproduce results
dvc repro

# Retrieve data e.g. images from data remote (without the need to reprocess)
dvc pull
```

The workflow assumes that the python environment as given in `environment.yaml` is installed and activated

```
mamba env create -f environment.yaml
```

## Labeling methods
### Brilouet et al. (2023) based on SAR backscatter
The method has been programmed following instructions given in the manuscript and is available at in [gradient_detection_SAR_brilouet_etal_2023.py](src/gradient_detection_SAR_brilouet_etal_2023.py). Figures given in the manuscript are reprodued and available on the [DVC Google Drive `figures` remote](https://drive.google.com/drive/folders/1va9TbLCB5q19ASfD0At3meXLl7xSUuI7).

## Contribution

> [!NOTE]
> The data remote is a AWS bucket that only allows annonymous reads. To push to the remote, access needs to be granted and the credentials be activated. The latter can be achieved by running `aws configure` and provide the credentials as given on the AWS webpage.
