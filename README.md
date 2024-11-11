# StREEQ

Numerical error estimation for stochastic and deterministic code responses.

## Usage

StREEQ has one main executable, the main script `streeq`. To use this executable, you need to ensure you have the correct environment loaded (see Environment section).
Run

```
$ streeq -h
```

to see available options.

## Versions

This project will use [Semantic Versioning](https://semver.org/).
The StREEQ development team will support the latest minor version series with bugfixes (e.g., once version `1.1.0` is
released, no bugfixes will be backported to the `1.0.z` series).

StREEQ development versions will be in the form `x.y_dev`, e.g. `0.11_dev` when the latest released version begins with
`0.11`.


### Branches

The following branches will be maintained in the repo:
* `master` -- stable version, points to latest release.
* `develop` -- latest development version.
* `releases/vX.Y` -- for example, `releases/v0.11`, `releases/v1.0`.  Contains a snapshot preparatory to releasing on
  this minor version series.

## Environment

StREEQ uses a conda environment to manage its dependencies. This section gives detail on what the environment contains, and how to set up your own if
needed.

### Dependencies

* `numpy` -- basic scientific computing tools
* `scipy` -- basic scientific computing tools
* `pyyaml` -- read and write YAML files
* `pandas` -- data analysis and manipulation
* `matplotlib` -- plotting
* `cvxopt` -- linear optimization
* `mpi4py` -- MPI execution
* `pytest` -- unit testing framework

### Setup

Edit `~/.condarc` to contain the following if you are not satisfied with the defaults:
```
envs_dirs:          # defaults to ~/.conda/envs
- /path/to/conda/envs
pkgs_dirs:          # defaults to ~/.conda/pkgs
- /path/to/conda/pkgs
```

Then run the following conda commands:
```
$ conda create -n streeq python=3.9.7
$ conda install -n streeq numpy scipy pyyaml pandas matplotlib cvxopt mpi4py pytest
```

## Running unit tests

StREEQ uses the [pytest](https://pytest.org) framework for unit testing sections of the code.

All unit tests can be executed by running the script `run_unit_tests`, located in the `tests` folder.
The script is set up to work correctly when run from any directory.
Note that a StREEQ environment must be loaded for the tests to run.

Some notes on organization:
* Unit tests are located in `tests/unit`
* The `Util` directory contains common unit test utilities, other directories contain tests and their associated
  datasets.

## Running regression tests

StREEQ uses [vvtest](https://github.com/sandialabs/vvtest) to run end-to-end regression tests of StREEQ.

The regression tests can be executed by navigating to `tests/regression` and running `vvtest`.
Note that this requires a StREEQ environment to be loaded and `vvtest` to be present in the `PATH`.
Both requirements are satisfied by loading any StREEQ module on Sandia computing resources.

## Release process

* Start a new `releases/vX.Y` branch from the tip of develop with the appropriate major/minor version number for the
  next release.
* Change the version number (`__version__` variable in `core/StREEQ.py`).
* If any bugs appear before the release, cherry-pick the bugfix commits from develop to this branch.
* Once the release is ready, merge it on to master, and tag the commit on the master branch with the version number in
  the format `vX.Y.Z`.
* If new bugfixes are required before the next minor release, cherry-pick them onto `releases/vX.Y`, increment the
  patch version (Z) by one, and merge to master, tagging as appropriate.
* Follow StREEQ Installer instruction to update the StREEQ module.
* After starting the `releases/vX.Y` branch, increment the version on the develop branch to `X.Y_dev`.

