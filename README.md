# PyEI

PyEI is a Python library for ecological inference. It is new and under active development, so expect rough edges and bugs -- and for additional features and documentation to be coming quickly!

## Want to use PyEI? Start here.

### Installation
You can install with pip:

```
pip install git+git://github.com/mggg/ecological-inference.git
```
### Example notebooks

Check out the [example notebooks](https://github.com/mggg/ecological-inference/tree/main/pyei/examples) for sample code
that shows how to run and adjust the various models in PyEI on datesets.  

For two-by-two cases, check out the examples in `santa_clara_demo.ipynb`.

For r-by-c cases, check out the examples in `santa_clara_demo_r_by_c.ipynb`.

For examples of more in depth model comparison and checking steps with PyEI, see `model_eval_and_comparison_demo.ipynb`.

### Issues

Feel free to file an issue if you are running into trouble or if there is a feature you'd particularly like to see, and we will do our best to get to it!


## Want to contribute to PyEI? Start here.

Contributions are welcome! 

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

### Testing

After making changes, make sure everything works by running

```bash
./scripts/lint_and_test.sh
```

This will also run automatically when you make a pull request, so if you have trouble getting that to run, just open the PR, and we can help!
