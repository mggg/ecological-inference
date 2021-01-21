"""Helpers for managing data files."""
from dataclasses import dataclass

import pandas as pd

__all__ = ["Datasets"]


@dataclass
class _DataSet:
    """Class to hold datasets and related information.

    TODO: Add description, provenance, and other metadata here.
    """

    url: str

    def to_dataframe(self) -> pd.DataFrame:
        """Materialize as a pandas dataframe."""
        return pd.read_csv(self.url)


class Datasets:  # pylint: disable=too-few-public-methods
    """Available datasets related to ecological inference.

    These support examples in the library. Please open an issue or pull request if you would
    like to see other specific examples, or have questions about these."""

    Santa_Clara = _DataSet(
        "https://raw.githubusercontent.com/gerrymandr/ei-app/master/santaClara.csv"
    )
    Waterbury = _DataSet("https://raw.githubusercontent.com/gerrymandr/ei-app/master/waterbury.csv")
