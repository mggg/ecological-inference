"""A package for rpv and ecological inference"""

__version__ = "1.1.4"

# Import main classes and functions
from .data import Datasets
from .goodmans_er import GoodmansER, GoodmansERBayes
from .plot_utils import (
    plot_boxplots,
    plot_kdes,
    plot_precinct_scatterplot,
    plot_summary,
)
from .r_by_c import RowByColumnEI
from .two_by_two import TwoByTwoEI, TwoByTwoEIBaseBayes

__all__ = [
    "Datasets",
    "GoodmansER",
    "GoodmansERBayes",
    "RowByColumnEI",
    "TwoByTwoEI",
    "TwoByTwoEIBaseBayes",
    "plot_boxplots",
    "plot_kdes",
    "plot_precinct_scatterplot",
    "plot_summary",
]
