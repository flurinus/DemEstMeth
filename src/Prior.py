'''
Created on Jun 27, 2014

@author: flurin
'''
import numpy as np
import scipy as sp
import multiprocessing as mp
from LausTTTData import TINFLinkCount
from functools import partial
import RouteChoice


class DemPrior:
    '''
    Contains historical information on pedestrian flows and train
    induced flows of pedestrians (Tinf), extrapolated from the timetable.
    '''

    def __init__(self, estim_param, network, data):
        '''
        Constructor. This constructor creates an instance of the prior.
        '''
        print "Predicting historical and schedule-based prior",
        
        self.tinf_distr = generateTinfMC(estim_param, network)
        self.tinf_mean = np.mean(self.tinf_distr, axis=1)
        self.tinf_std = np.std(self.tinf_distr, axis=1)
                        
        dest_flows = RouteChoice.generate_dest_flows(estim_param, network, data)
        M_sales, v_sales = RouteChoice.sales_data_GLS(estim_param, network, dest_flows)
        M_depflow, v_depflow = RouteChoice.platform_centroid_dep_flow(estim_param, network, dest_flows)
        M_rsplits, v_rsplits = RouteChoice.route_split_GLS(estim_param, network, data, dest_flows)
        M_agg_rsplits, v_agg_rsplits = RouteChoice.agg_cum_route_split_GLS(estim_param, network, data, dest_flows)
                 
        self.M = np.concatenate((M_sales, M_depflow, M_rsplits), axis=0)
        self.v = np.concatenate((v_sales, v_depflow, v_rsplits), axis=0)
        self.Magg = np.concatenate((M_sales, M_depflow, M_agg_rsplits), axis=0)
        self.vagg = np.concatenate((v_sales, v_depflow, v_agg_rsplits), axis=0)
        estim_param.print_incr_runtime()

        
def generateTinfMC(estim_param, network):
    '''
    Generates the train induced flows using as many processors as specified the main class.
    '''
    tinfMC = sp.zeros((len(network.edges_TINF_dict) * len(estim_param.tint_dict), estim_param.MC_iterations))
    poolTINF = mp.Pool(estim_param.parallel_threads)
    for i in range(estim_param.MC_iterations):
        callbackTINF = partial(multiproc_callback_MC, mat=tinfMC, col_ind=i)
        poolTINF.apply_async(instantiateTinf, args=(estim_param, network), callback=callbackTINF)
    poolTINF.close()
    poolTINF.join()
    return tinfMC

        
def instantiateTinf(estim_param, network):
    '''
    Creates one instance of the train induced flows.
    '''
    tinfEstimate = TINFLinkCount(estim_param, network)
    return tinfEstimate.TINF
        
    
def multiproc_callback_MC(res, mat, col_ind):
    '''
    Stores the result of an asynchronous computation in a matrix
    '''
    mat[:, col_ind] = res
