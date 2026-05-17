"""A package for rpv and ecological inference"""

__version__ = "1.1.4"

from .data import Datasets
from .goodmans_er import GoodmansER, GoodmansERBayes
from .plot_utils import (
    plot_boxplots,
    plot_conf_or_credible_interval,
    plot_intervals_all_precincts,
    plot_kdes,
    plot_margin_kde,
    plot_polarization_kde,
    plot_precinct_scatterplot,
    plot_precincts,
    plot_summary,
    tomography_plot,
)
from .r_by_c import RowByColumnEI
from .two_by_two import (
    TwoByTwoEI,
    TwoByTwoEIBaseBayes,
    ei_beta_binom_model,
    ei_beta_binom_model_modified,
)

__all__ = [
    "Datasets",
    "GoodmansER",
    "GoodmansERBayes",
    "RowByColumnEI",
    "TwoByTwoEI",
    "TwoByTwoEIBaseBayes",
    "ei_beta_binom_model",
    "ei_beta_binom_model_modified",
    "plot_boxplots",
    "plot_conf_or_credible_interval",
    "plot_intervals_all_precincts",
    "plot_kdes",
    "plot_margin_kde",
    "plot_polarization_kde",
    "plot_precinct_scatterplot",
    "plot_precincts",
    "plot_summary",
    "tomography_plot",
]
