import xarray as xr
import arviz as az
from pyei.two_by_two import TwoByTwoEI, TwoByTwoEIBaseBayes
from pyei.r_by_c import RowByColumnEI


def to_netcdf(ei_object, filepath):
    """
    Saves traces and some metadata for EI objects to disk

    Parameters:
    -----------
    ei_object: an object of class TwoByTwoBaseBayes or RowByColumnEI
    filepath : str
    The path to the file where the data will be saved
    Notes:
    ------
    
    """
    # Check if model has been fit yet
    # Add properties as attrs to sim_trace
    # if not (isinstance(ei_object, TwoByTwoEIBaseBayes) or isinstance(ei_object, RowByColumnEI)):
    #     raise ValueError("ei_object must be of class TwoByTwoEIBaseBayes or RowByColumnEI")

    is_two_by_two = isinstance(ei_object, TwoByTwoEIBaseBayes) #bool

    if is_two_by_two:
        attr_list = ['model_name','precinct_pops', 'precinct_names', 'demographic_group_name', 'candidate_name',
        'demographic_group_fraction', 'votes_fraction']
        ei_object.sim_trace.posterior.attrs['is_two_by_two'] = 'true' # store whether 2 by 2 or r by c

    else: # r by c
        #attr_list=[]
        attr_list = ['model_name', 'precinct_pops', 'precinct_names', 'demographic_group_names','candidate_names', 'num_groups_and_num_candidates']
        ei_object.sim_trace.posterior.attrs['is_two_by_two'] = 'false'
                
    for attr in attr_list:
        if getattr(ei_object, attr) is not None:
            ei_object.sim_trace.posterior.attrs[attr] = getattr(ei_object, attr)
    
    #Use az.InferenceData's to_netcdf
    ei_object.sim_trace.to_netcdf(filepath)

    if not is_two_by_two:
        mode = "a"
        for attr in ['demographic_group_fractions', 'votes_fractions']: #array atts
            data = xr.DataArray(getattr(ei_object, attr), name=attr)
            data.to_netcdf(filepath, mode=mode, group=attr)
            data.close()


def from_netcdf(filepath):
    """
    Loads traces and metadata for EI objects to disk

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
    idata = az.from_netcdf(filepath)

    attrs_dict = idata.posterior.attrs
    attr_list = list(idata.posterior.attrs.keys())
    attr_list.remove('created_at')
    attr_list.remove('arviz_version')

    if attrs_dict['is_two_by_two']=='true':
        ei = TwoByTwoEI(attrs_dict['model_name']) #initialize EI object
        ei.calculate_sampled_voting_prefs() # calculate polity-wide samples
    else: #otherwise it's an R by C
        ei = RowByColumnEI(attrs_dict['model_name'])

        ei.demographic_group_fractions = idata.demographic_group_fractions['demographic_group_fractions'].to_numpy()
        del(idata.demographic_group_fractions)
        ei.votes_fractions = idata.votes_fractions['votes_fractions'].to_numpy()
        del(idata.votes_fractions)

    for attr in attr_list: # set attrs of the EI object
        setattr(ei, attr, attrs_dict[attr])
        del idata.posterior.attrs[attr] # these vars only attached to the posterior for saving/loading
    
    ei.sim_trace = idata
    ei.calculate_summary() # calculate summary quantities

    return ei