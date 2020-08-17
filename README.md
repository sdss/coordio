# coordIO

[![Versions](https://img.shields.io/pypi/pyversions/sdss-coordio)](https://docs.python.org/3/)
[![Documentation Status](https://readthedocs.org/projects/sdss-coordio/badge/?version=latest)](https://sdss-coordio.readthedocs.io/en/latest/?badge=latest)
[![Test](https://img.shields.io/github/workflow/status/sdss/coordio/Test)](https://github.com/sdss/coordio/actions)
[![Coverage Status](https://codecov.io/gh/sdss/coordio/branch/master/graph/badge.svg)](https://codecov.io/gh/sdss/coordio)


Coordinate conversion for SDSS-V.


## Installation

In general you should be able to install ``coordio`` by doing

```console
pip install sdss-coordio
```

To build from source, use

```console
git clone git@github.com:sdss/coordio
cd coordio
pip install .[docs]
```

## Development

`coordio` uses [poetry](http://poetry.eustace.io/) for dependency management and packaging. To work with an editable install it's recommended that you setup `poetry` and install `coordio` in a virtual environment by doing

```console
poetry install
```

Pip does not support editable installs with PEP-517 yet. That means that running `pip install -e .` will fail because `poetry` doesn't use a `setup.py` file. As a workaround, you can use the `create_setup.py` file to generate a temporary `setup.py` file. To install `coordio` in editable mode without `poetry`, do

```console
pip install --pre poetry
python create_setup.py
pip install -e .
```
