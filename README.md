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

## Contribution

> [!NOTE]
> The data remote is a AWS bucket that only allows annonymous reads. To push to the remote, access needs to be granted and the credentials be activated. The latter can be achieved by running `aws configure` and provide the credentials as given on the AWS webpage.