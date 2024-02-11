# PyEI

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03397/status.svg)](https://doi.org/10.21105/joss.03397)

PyEI is a Python library for ecological inference. The target audience is the analyst with an interest in the phenomenon called Racially Polarized Voting.

Racially Polarized Voting is a legal concept developed through case law under the Voting Rights Act of 1965; its genesis is in the majority opinion of ***Thornburg v. Gingles (1982)***. Considered the “evidentiary linchpin” for vote dilution cases, RPV is a necessary, but not sufficient, condition that plaintiffs must satisfy for a valid claim. 

Toward that end, ecological inference uses observed data (historical election results), pairing voting outcomes with demographic information
for each precinct in a given polity, to infer voting patterns for each demographic group.

PyEI brings together a variety of ecological inference methods in one place and facilitates reporting and plotting results; quantifying the uncertainty associated with results under a given model; making comparisons between methods; and bringing relevant diagnostic tools to bear on ecological inference methods.

PyEI is relatively new and under active development, so expect rough edges and bugs -- and for additional features and documentation to be coming quickly!

## Want to use PyEI? Start here.

### Installation
You can install the latest release from `PyPi` with:

```
pip install pyei
```

Or, install directly from GitHub for the most up-to-date (but potentially less stable) version:

```
pip install git+https://github.com/mggg/ecological-inference.git
 ```
 
If you would like to explore PyEI without installation, you can explore this [interactive Colab notebook](https://colab.research.google.com/drive/1Vr1kKAAHgdcUhPrpFsYc1Kz31nbcpZjP#scrollTo=_ASEm5L3UUAS) (just note that inference might be slow!)


### Example notebooks

Check out the [intro notebooks](https://github.com/mggg/ecological-inference/tree/main/pyei/intro_notebooks) and [example notebooks](https://github.com/mggg/ecological-inference/tree/main/pyei/examples) for sample code
that shows how to run and adjust the various models in PyEI on datesets.  

If you are new to ecological inference generally, start with [`pyei/intro_notebooks/Introduction_toEI.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/intro_notebooks/Introduction_to_EI.ipynb).

If you are familiar with ecological inference and want an overview of PyEI and how to use it (with examples), then start with [`intro_notebooks/PyEI_overview.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/intro_notebooks/PyEI_overview.ipynb).

To explore EI's plotting functionality, check out [`pyei/intro_notebooks/Plotting_with_PyEI.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/intro_notebooks/Plotting_with_PyEI.ipynb).

For more work with two-by-two examples, see in [`pyei/examples/santa_clara_demo.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/examples/santa_clara_demo.ipynb).

For more work with r-by-c examples, see [`pyei/examples/santa_clara_demo_r_by_c.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/examples/santa_clara_demo_r_by_c.ipynb).

For examples of model comparison and checking steps with PyEI, see [`pyei/examples/model_eval_and_comparison_demo.ipynb`](https://github.com/mggg/ecological-inference/blob/main/pyei/examples/model_eval_and_comparison_demo.ipynb).

### Issues

Feel free to file an issue if you are running into trouble or if there is a feature you'd particularly like to see, and we will do our best to get to it!


## Want to contribute to PyEI? Start here.

Contributions are welcome! 

Uses Python 3.10. After cloning the environment, you should be able to use either `virtualenv` or `conda` to run the code. The second (`conda`) is probably easier for development, but `virtualenv` is used for the project's CI.

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
conda create --name pyei --channel conda-forge python=3.10 --file requirements.txt --file requirements-dev.txt # create conda environment and install requirements
conda activate pyei
pip install -e . #install project locally
```

### Testing

After making changes, make sure everything works by running

```bash
./scripts/lint_and_test.sh
```

This will also run automatically when you make a pull request, so if you have trouble getting that to run, just open the PR, and we can help!


## Citation

If you are using PyEI, please cite it as: 

Knudson et al., (2021). PyEI: A Python package for ecological inference. Journal of Open Source Software, 6(64), 3397, https://doi.org/10.21105/joss.03397

BibTeX:

```
@article{Knudson2021,
  doi = {10.21105/joss.03397},
  url = {https://doi.org/10.21105/joss.03397},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {64},
  pages = {3397},
  author = {Karin C. Knudson and Gabe Schoenbach and Amariah Becker},
  title = {PyEI: A Python package for ecological inference},
  journal = {Journal of Open Source Software}
}
```


