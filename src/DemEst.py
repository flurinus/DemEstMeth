'''
Created on Mar 24, 2014

@author: eduard, flurin, quentin
'''

import numpy as np
import scipy.optimize as opti
import os.path

import multiprocessing as mp
from functools import partial
from Prior import multiproc_callback_MC


class DemandEstimation:
    '''
    Creates the route demand in the network.
    '''
    def __init__(self, estim_param, network, assignment, data, prior):
        '''
        Constructor
        '''
        # create precomputed directory if it does not exist
        if not os.path.exists(estim_param.path_precomp):
            os.makedirs(estim_param.path_precomp)
        
        self.demand_distr = np.zeros((len(estim_param.tint_dict) * len(network.routes_dict), estim_param.MC_iterations))
        
        if os.path.isfile(estim_param.path_precomputed_dem + "_demand_distr.csv") and estim_param.forceRecompDemand is False:
    
            print "Loading pre-computed demand",
            self.demand_distr = np.loadtxt(estim_param.path_precomputed_dem + "_demand_distr.csv", delimiter=",")
            
        else:
            print "Estimating demand",
            
            if estim_param.MC_iterations == 1:
                if estim_param.muTINF > 0.0:
                    print "Warning: Single estimation only."
                
                self.demand_distr = compute_route_demand(estim_param, network, assignment, data, prior, 0)
                
            else:
                poolDemEst = mp.Pool(estim_param.parallel_threads)
                for i in range(estim_param.MC_iterations):
                    callbackDemEst = partial(multiproc_callback_MC, mat=self.demand_distr, col_ind=i)
                                        
                    poolDemEst.apply_async(compute_route_demand, args=(estim_param, network, assignment, data, prior, i), callback=callbackDemEst)
                    
                poolDemEst.close()
                poolDemEst.join()
                           
            np.savetxt(estim_param.path_precomputed_dem + "_demand_distr.csv", self.demand_distr, delimiter=",")
            
        estim_param.print_incr_runtime()

    def add_noise(self):
        '''
        For debugging only. add unbiased noise; prevent negative demand
        '''
        for i in range(self.demand_distr.shape[0]):
            for j in range(self.demand_distr.shape[1]):
                noise = np.random.normal(0, 1)
                if abs(noise) < abs(self.demand_distr[i][j]):
                    self.demand_distr[i][j] = self.demand_distr[i][j] + noise

                    
def compute_route_demand(estim_param, network, assignment, data, prior, tinf_instance):
    '''
    Computes the demand on each route based on the assiment matrix and prior
    '''
    muZero = estim_param.muZero
    muLink = estim_param.muLink
    muSubRoute = estim_param.muSubRoute
    muTINF = estim_param.muTINF
    muHist = estim_param.muHist
    muHistAgg = estim_param.muHistAgg
        
    concat_matrix = None
    concat_obj = None
    
    if muZero > 0:
            concat_matrix = muZero * np.eye(len(network.routes_dict) * len(estim_param.tint_dict))
            concat_obj = np.zeros(len(network.routes_dict) * len(estim_param.tint_dict))

    if muLink > 0:
        link_flows = data.f_hat
        assg_mat_link_flows = assignment.linkFlowAssgn_prime

        if link_flows is None or assg_mat_link_flows is None:
            print "Warning: Link flows and/or assignment not specified."
        elif concat_matrix is None:
            concat_matrix = muLink * assg_mat_link_flows
            concat_obj = muLink * np.array(link_flows)
        else:
            concat_matrix = np.concatenate((concat_matrix, muLink * assg_mat_link_flows), axis=0)
            concat_obj = np.concatenate((concat_obj, muLink * np.array(link_flows)), axis=0)

    if muSubRoute > 0:
        subroute_flows = data.g_hat
        assg_mat_subroute_flows = assignment.subRouteAssgn_prime
        
        if subroute_flows is None or assg_mat_subroute_flows is None:
            print "Warning: Subroute flows and/or assignment not specified."
        elif concat_matrix is None:
            concat_matrix = muSubRoute * assg_mat_subroute_flows
            concat_obj = muSubRoute * np.array(subroute_flows)
        else:
            concat_matrix = np.concatenate((concat_matrix, muSubRoute * assg_mat_subroute_flows), axis=0)
            concat_obj = np.concatenate((concat_obj, muSubRoute * np.array(subroute_flows)), axis=0)
            
    if muTINF > 0:
        assg_mat_tinf_flows = assignment.accAssgn_prime
        tinf_flows = prior.tinf_distr[:, tinf_instance]

        if tinf_flows is None or assg_mat_tinf_flows is None:
            print "Warning: Train-induced flows and/or assignment not specified."
        elif concat_matrix is None:
            concat_matrix = muTINF * assg_mat_tinf_flows
            concat_obj = muTINF * np.array(tinf_flows)
        else:
            concat_matrix = np.concatenate((concat_matrix, muTINF * assg_mat_tinf_flows), axis=0)
            concat_obj = np.concatenate((concat_obj, muTINF * np.array(tinf_flows)), axis=0)
    
    if muHist > 0:
        histPriorMat = prior.M
        histPriorVec = prior.v
        
        if histPriorVec is None or histPriorMat is None:
            print "Warning: Historical prior not specified."
        elif concat_matrix is None:
            concat_matrix = muHist * histPriorMat
            concat_obj = muHist * np.array(histPriorVec)
        else:
            concat_matrix = np.concatenate((concat_matrix, muHist * histPriorMat), axis=0)
            concat_obj = np.concatenate((concat_obj, muHist * np.array(histPriorVec)), axis=0)
    
    if muHistAgg > 0:
        histPriorAggMat = prior.Magg
        histPriorAggVec = prior.vagg
        
        if histPriorAggVec is None or histPriorAggMat is None:
            print "Warning: Aggregate historical prior not specified."
        elif concat_matrix is None:
            concat_matrix = muHistAgg * histPriorAggMat
            concat_obj = muHistAgg * np.array(histPriorAggVec)
        else:
            concat_matrix = np.concatenate((concat_matrix, muHistAgg * histPriorAggMat), axis=0)
            concat_obj = np.concatenate((concat_obj, muHistAgg * np.array(histPriorAggVec)), axis=0)

    #return bfgs.nls_lbfgs_b(concat_matrix, concat_obj)
    #return sgd.stochastic_gradient_descent(concat_obj, concat_matrix, verbose=False)
    #return active.nls_activeset(concat_matrix, concat_obj, None)
    #return ls.lsqnonneg(concat_matrix, concat_obj, opts = {'show_progress': False})
    #return pg.nls_projgrad(concat_matrix, concat_obj)
    
    return opti.nnls(concat_matrix, concat_obj)[0]

