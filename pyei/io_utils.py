"""Utility functions for saving and loading ei objects to and from disk."""

from typing import Any, cast

import arviz as az
import xarray as xr

from pyei.r_by_c import RowByColumnEI
from pyei.two_by_two import TwoByTwoEI, TwoByTwoEIBaseBayes


def to_netcdf(ei_object: TwoByTwoEIBaseBayes | RowByColumnEI, filepath: str) -> None:
    """Saves traces and some metadata for EI objects to disk.

    Parameters:
    -----------
    ei_object: an object of class TwoByTwoBaseBayes or RowByColumnEI
    filepath : str
    The path to the file where the data will be saved
    """
    # Check if model has been fit yet
    # Add properties as attrs to sim_trace
    # if not (isinstance(ei_object, TwoByTwoEIBaseBayes) or isinstance(ei_object, RowByColumnEI)):
    #     raise ValueError("ei_object must be of class TwoByTwoEIBaseBayes or RowByColumnEI")

    if ei_object.sim_trace is None:
        raise ValueError("ei_object must be fit before saving")

    # arviz InferenceData uses dynamic attribute access for groups (e.g. .posterior)
    sim_trace = cast(Any, ei_object.sim_trace)
    is_two_by_two = isinstance(ei_object, TwoByTwoEIBaseBayes)  # bool

    if is_two_by_two:
        attr_list = [
            "model_name",
            "precinct_pops",
            "precinct_names",
            "demographic_group_name",
            "candidate_name",
            "demographic_group_fraction",
            "votes_fraction",
        ]
        sim_trace.posterior.attrs["is_two_by_two"] = (
            "true"  # store whether 2 by 2 or r by c
        )

    else:  # r by c
        # attr_list=[]
        attr_list = [
            "model_name",
            "precinct_pops",
            "precinct_names",
            "demographic_group_names",
            "candidate_names",
            "num_groups_and_num_candidates",
        ]
        sim_trace.posterior.attrs["is_two_by_two"] = "false"

    for attr in attr_list:
        if getattr(ei_object, attr) is not None:
            sim_trace.posterior.attrs[attr] = getattr(ei_object, attr)

    # Use az.InferenceData's to_netcdf
    sim_trace.to_netcdf(filepath)

    if not is_two_by_two:
        for attr in ["demographic_group_fractions", "votes_fractions"]:  # array atts
            data = xr.DataArray(getattr(ei_object, attr), name=attr)
            data.load()
            data.to_netcdf(filepath, mode="a", group=attr, engine="netcdf4")
            data.close()


def from_netcdf(filepath: str) -> TwoByTwoEI | RowByColumnEI:
    """Loads traces and metadata for EI objects to disk

    Parameters
    ----------
    filepath : str
    The path to the file from which the data will loaded

    Returns:
    --------
    ei: an object of type TwoByTwoEI or RowByColumnEI
    with sim_trace and most other atrributes set as they would
    be when fit. Note sim_model is not saved/loaded
    """
    # arviz InferenceData accesses groups (.posterior, .demographic_group_fractions,
    # .votes_fractions) dynamically — not visible to the stubs.
    idata = cast(Any, az.from_netcdf(filepath, engine="netcdf4"))

    attrs_dict = idata.posterior.attrs
    attr_list = list(idata.posterior.attrs.keys())
    attr_list.remove("created_at")
    attr_list.remove("arviz_version")

    ei_object: TwoByTwoEI | RowByColumnEI
    is_two_by_two = attrs_dict["is_two_by_two"] == "true"
    if is_two_by_two:
        ei_object = TwoByTwoEI(attrs_dict["model_name"])
    else:
        ei_object = RowByColumnEI(attrs_dict["model_name"])

        ei_object.demographic_group_fractions = idata.demographic_group_fractions[
            "demographic_group_fractions"
        ].to_numpy()
        del idata.demographic_group_fractions
        ei_object.votes_fractions = idata.votes_fractions["votes_fractions"].to_numpy()
        del idata.votes_fractions

    for attr in attr_list:  # set attrs of the EI object
        setattr(ei_object, attr, attrs_dict[attr])
        del idata.posterior.attrs[
            attr
        ]  # these vars only attached to the posterior for saving/loading

    ei_object.sim_trace = idata
    if is_two_by_two:
        # 2x2 summary reads from sampled_voting_prefs, which must be computed
        # from the freshly-restored sim_trace first.
        cast(TwoByTwoEI, ei_object).calculate_sampled_voting_prefs()
    ei_object.calculate_summary()

    return ei_object
