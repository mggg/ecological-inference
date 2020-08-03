# pyei

## Getting started

Uses Python>=3.7. After cloning the environment, you should be able to use either `virtualenv` or `conda` to run the code. The second (`conda`) is probably easier for development, but `virtualenv` is used for the project's CI.

Here is how to create and activate each environment. See the docs for more elaborate details:

### Install with virtualenv

```bash
virtualenv pyei_venv           # create virtualenv
source pyei_venv/bin/activate  # activate virtualenv
python -m pip install -U pip   # upgrade pip
python -m pip install -e .     # install project locally
python -m pip install -r requirements-dev.txt  # install dev requirements
```

### Install with conda

```bash
conda create --name pyei python=3.7  # create conda env with python 3.7
source activate pyei                 # activate conda env
# See requirements.txt and requirements-dev.txt
conda install pymc3 mkl-service scikit-learn matplotlib seaborn black mypy pylint pytest pytest-cov
```

## Contributing

Contributions are welcome! See the `Getting started` section for installing the project. After making changes, make sure everything works by running

```bash
./scripts/lint_and_test.sh
```

This will also run automatically when you make a pull request, so if you have trouble getting that to run, just open the PR, and we can help!
