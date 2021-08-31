# TODO

* General
    - [x] Setup up with conda
    - [x] Pull labels into repo
    - [x] CLI
    - [ ] Documentation: 80% done
    - [ ] Even more aggressive modularization
    - [x] Add preprocessing shell script and raw data
    - [ ] Verify experiments fully reproducible: old numbers roughly reproducible, though now new, better performing and reproducible numbers 
    - [ ] Pull in Leiden's analysis code
    - [ ] Test on Windows
    - [ ] Snapshot of data (in particular proposals/final acts before and after preprocessing validating our approach)

* Once final acts annotations are available
    - [ ] Migrate transformers pipeline
    - [x] Comparatively evaluate generalization of scaling from LP-only to LP+FA

# OEIL

This repository implements the scaling algorithms for "PAPER"
 
**Please note** that this project has only been tested on Linux so far and should run seamlessly on MacOS as well. Windows support remains to be tested.

This work is part of the research collaboration "EU In Action" of [University of Leiden](https://www.universiteitleiden.nl/en), [University of Strathclyde](https://www.strath.ac.uk/), and [University of Mannheim]() funded by [Norface](https://www.norface.net/).

More information can be found on the [project homepage](https://www.euinaction.eu/).


## Installation

Prior to usage, please install an conda distribution for your operation system. Instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Once conda is readily available in your `PATH`, go to the project folder an run: `conda env create -f environment.yml`

## Preprocessing

On Unix systems, it simply suffices to run `prepare.sh` prior to training a model which automatically runs the steps as laid out for Windows:

1. Extract `./observatory_summaries.zip` into `./data/summaries`
2. Run `python preprocess.py`
3. Run `python learn_tokenizer.json` (though a pre-trained tokenizer is provided)

## Usage

The experiments are configured using [Hydra](https://hydra.cc/) for which the respective configuration files can be found in `$PROJECT/configs`, which has the following structure:

```
├── config.yaml
├── experiment
│   └── lr.yaml
├── hparams
│   └── lr.yaml
├── __init__.py
└── model
    └── lr.yaml
```
* `defaults`: `config.yaml` specifies the global default configuration and is discouraged to be modified
* `hparams`:  stores the hyperparameters for a model (which might naturally differ by model or group of models)
* `model`:  comprises the base configuration per model (class); for instance, any

### Pipeline

Hydra calls the `_target_` function pointed to in the `experiment` configuration which denotes the pipeline for any group of classifiers.

For instance, other [scikit-learn](https://scikit-learn.org/) can be naturally ran by reconfiguring `model/lr.yaml` accordingly for a different scikit-learn classifier.

### Reproduction

Reproducing our results denotes running our provided `experiment` configurations, for instance, for the logistic regression classifier:

`python run.py experiment=lr`

You can overwrite single parameters of your experiment like so:

`python run.py experiment=lr hparams.cv.scoring="f1_weighted"`

See [Hydra](https://hydra.cc/) for more information on how to use the commandline-interface.

#### Low-level details

The fully documented scikit-learn pipeline can be found at `./src/run/classifier.py`.

# Pull Requests

PRs are very welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b my_contribution`)
3. Make your changes
4. Stash and commit your Changes (`git add -u && git commit -m 'Add my amazing contribution'`)
5. Push to the Branch (`git push origin my_contribution`)
6. Open a Pull Request by going to the project webpage
