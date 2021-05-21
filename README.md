# PyEI

PyEI is a Python library for ecological inference. It is new and under active development, so expect rough edges and bugs -- and for additional features and documentation to be coming quickly!

## Want to use PyEI? Start here.

### Installation
You can install with pip:

```
pip install git+git://github.com/mggg/ecological-inference.git
```
### Example notebooks

Check out the [intro notebooks](https://github.com/mggg/ecological-inference/tree/main/pyei/intro_notebooks) and [example notebooks](https://github.com/mggg/ecological-inference/tree/main/pyei/examples) for sample code
that shows how to run and adjust the various models in PyEI on datesets.  

If you are new to ecological inference generally, start with `intro_notebooks/Introduction_toEI.ipynb`.

If you are familiar with ecological inference and want an overview of PyEI and how to use it, with examples start with `intro_notebooks/PyEI_overview.ipynb`.

To explore EI's plotting functionality, check out `intro_notebooks/Plotting_with_PyEI.ipynb`.

For more work with two-by-two examples, see in `examples/santa_clara_demo.ipynb`.

For more work with r-by-c examples, see `examples/santa_clara_demo_r_by_c.ipynb`.

For examples of depth model comparison and checking steps with PyEI, see `examples/model_eval_and_comparison_demo.ipynb`.

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
conda create --name pyei python=3.8  # create conda env with python 3.8
conda activate pyei                 # activate conda env
# See requirements.txt and requirements-dev.txt
conda install pip
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-dev.txt
pip install -e . 
```

### Testing

After making changes, make sure everything works by running

```bash
./scripts/lint_and_test.sh
```

This will also run automatically when you make a pull request, so if you have trouble getting that to run, just open the PR, and we can help!
