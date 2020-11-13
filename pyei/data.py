"""Helpers for managing data files."""
import io
import os
import pkgutil

import pandas as pd

__all__ = ["get_data"]


def get_data(filename):
    """Returns a dataframe for example datasets.

    Will either load remotely, or locally. Currently supports:

    - santaClara.csv
    - waterbury.csv

    TODO: Add details about these datasets.

    Parameters
    ----------
    filename: str
        File to load. See above for valid files.

    Returns
    -------
    BytesIO of the data
    """
    if filename == "santaClara.csv":
        return pd.read_csv(
            "https://raw.githubusercontent.com/gerrymandr/ei-app/master/santaClara.csv"
        )
    elif filename == "waterbury.csv":
        return pd.read_csv(
            "https://raw.githubusercontent.com/gerrymandr/ei-app/master/waterbury.csv"
        )
    else:
        raise ValueError('''get_data() currently only supports filenames "santaClara.csv" or "waterbury.csv".
        Use, e.g., pandas.read_csv()" if you'd like to load your own data file''')

    # This does not work yet (9/2/20), but will collect
    # files checked into pyei/examples/data/<filename>
    # else:
    #    data_pkg = "pyei.examples"
    #    return pd.read_csv(io.BytesIO(pkgutil.get_data(data_pkg, os.path.join("data", filename))))
