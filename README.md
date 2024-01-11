# Cold pool detection
This repository is work in progress.

## Reproducability / Usage
This repository uses [dvc](dvc.org) to keep track of workflows and data.

```bash
# Reproduce results
dvc repro

# Retrieve data e.g. images from data remote
dvc pull
```

The workflow assumes that the python environment as given in `environment.yaml` is installed and activated

```
mamba env create -f environment.yaml
```