'''
Created on Jul 20, 2014

@author: flurin
'''

import numpy as np
from Eval import aggreg_time, aggreg_space
from Eval import error_distances


class EvaluationMD:
    '''
    Multi day version of Evaluation
    '''
    def __init__(self, estim_param, network, eval_set):
        print "Evaluating demand of", estim_param.case_study_name,
        
        for date in estim_param.date_set:
            if date == estim_param.date_set[0]:
                
                self.demand_dist = eval_set[date].demand_dist
                self.flow_dist = eval_set[date].flow_dist
                self.occ_est_dist = eval_set[date].occ_est_dist
                
                self.sub_dem_est_dist = eval_set[date].sub_dem_est_dist
                self.sub_dem_meas_dist = eval_set[date].sub_dem_meas_mean
                
                self.ASE_flow_meas_dist = eval_set[date].ASE_flow_meas_mean
                self.TINF_flow_prior_dist = eval_set[date].TINF_flow_prior_dist
                self.VS_inflow_meas_dist = eval_set[date].VS_inflow_meas_mean
                self.VS_outflow_approx_meas_dist = eval_set[date].VS_outflow_approx_meas_mean
                
                self.occ_meas_dist = eval_set[date].occ_meas_mean
                
            else:
                self.demand_dist = np.column_stack((self.demand_dist, eval_set[date].demand_dist))
                self.flow_dist = np.column_stack((self.flow_dist, eval_set[date].flow_dist))
                self.occ_est_dist = np.column_stack((self.occ_est_dist, eval_set[date].occ_est_dist))
                
                self.sub_dem_est_dist = np.column_stack((self.sub_dem_est_dist, eval_set[date].sub_dem_est_dist))
                self.sub_dem_meas_dist = np.column_stack((self.sub_dem_meas_dist, eval_set[date].sub_dem_meas_mean))
                
                self.ASE_flow_meas_dist = np.column_stack((self.ASE_flow_meas_dist, eval_set[date].ASE_flow_meas_mean))
                self.TINF_flow_prior_dist = np.column_stack((self.TINF_flow_prior_dist, eval_set[date].TINF_flow_prior_dist))
                self.VS_inflow_meas_dist = np.column_stack((self.VS_inflow_meas_dist, eval_set[date].VS_inflow_meas_mean))
                self.VS_outflow_approx_meas_dist = np.column_stack((self.VS_outflow_approx_meas_dist, eval_set[date].VS_outflow_approx_meas_mean))
                
                self.occ_meas_dist = np.column_stack((self.occ_meas_dist, eval_set[date].occ_meas_mean))
        
        # compute mean and std of estimated demand, flow and occupation
        self.demand_mean, self.demand_std = mean_std(self.demand_dist)
        self.demand_agg_time_mean, self.demand_agg_time_std = aggreg_time(self.demand_dist, estim_param.tint_eval_dict.keys(), network.routes_dict)
        self.demand_agg_space_mean, self.demand_agg_space_std = aggreg_space(self.demand_dist, estim_param.tint_eval_dict.keys(), network.routes_dict)
        
        self.flow_mean, self.flow_std = mean_std(self.flow_dist)
        self.flow_agg_time_mean, self.flow_agg_time_std = aggreg_time(self.flow_dist, estim_param.tint_eval_dict.keys(), network.edges_dict)
        
        self.occ_est_mean, self.occ_est_std = mean_std(self.occ_est_dist)
        self.occ_est_agg_space_mean, self.occ_est_agg_space_std = aggreg_space(self.occ_est_dist, estim_param.tint_eval_dict.keys(), network.areas_dict)
        
        self.occ_meas_mean, self.occ_meas_std = mean_std(self.occ_meas_dist)
        self.occ_meas_agg_space_mean, self.occ_meas_agg_space_std = aggreg_space(self.occ_meas_dist, estim_param.tint_eval_dict.keys(), network.areas_dict)
                
        self.sub_dem_est_mean, self.sub_dem_est_std = mean_std(self.sub_dem_est_dist)
        self.sub_dem_est_agg_time_mean, self.sub_dem_est_agg_time_std = aggreg_time(self.sub_dem_est_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        self.sub_dem_est_agg_space_mean, self.sub_dem_est_agg_space_std = aggreg_space(self.sub_dem_est_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        
        self.sub_dem_meas_mean, self.sub_dem_meas_std = mean_std(self.sub_dem_meas_dist)
        
        self.sub_dem_meas_agg_time_mean, self.sub_dem_meas_agg_time_std = aggreg_time(self.sub_dem_meas_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        self.sub_dem_meas_agg_space_mean, self.sub_dem_meas_agg_space_std = aggreg_space(self.sub_dem_meas_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
                
        self.ASE_flow_est_mean, self.ASE_flow_est_std = self.getFlowSubVector(estim_param, network, network.edges_ASE_dict)
        self.ASE_flow_meas_mean, self.ASE_flow_meas_std = mean_std(self.ASE_flow_meas_dist)
                
        self.TINF_flow_est_mean, self.TINF_flow_est_std = self.getFlowSubVector(estim_param, network, network.edges_TINF_dict)
        self.TINF_flow_prior_mean, self.TINF_flow_prior_std = mean_std(self.TINF_flow_prior_dist)
                
        self.VS_inflow_meas_mean, self.VS_inflow_meas_std = mean_std(self.VS_inflow_meas_dist)
        self.VS_inflow_est_mean, self.VS_inflow_est_std = self.getFlowSubVector(estim_param, network, network.VS_inflow_edges_dict)
        
        self.VS_outflow_approx_meas_mean, self.VS_outflow_approx_meas_std = mean_std(self.VS_outflow_approx_meas_dist)
        self.VS_outflow_est_mean, self.VS_outflow_est_std = self.getFlowSubVector(estim_param, network, network.VS_outflow_edges_dict)

        # est <-> meas errors
        self.occ_agg_space_err = error_distances(self.occ_meas_agg_space_mean, self.occ_est_agg_space_mean)
        self.subroute_err = error_distances(self.sub_dem_meas_mean, self.sub_dem_est_mean)
        self.subroute_agg_space_err = error_distances(self.sub_dem_meas_agg_space_mean, self.sub_dem_est_agg_space_mean)
        self.subroute_agg_time_err = error_distances(self.sub_dem_meas_agg_time_mean, self.sub_dem_est_agg_time_mean)
        
        self.ASE_flow_error = error_distances(self.ASE_flow_meas_mean, self.ASE_flow_est_mean)
    
        self.TINF_flow_error = error_distances(self.TINF_flow_prior_mean, self.TINF_flow_est_mean)
        
        self.VS_err_inflow = error_distances(self.VS_inflow_meas_mean, self.VS_inflow_est_mean)
        self.VS_approx_err_outflow = error_distances(self.VS_outflow_approx_meas_mean, self.VS_outflow_est_mean)

        # axis ticks for temporal plots
        self.axis_ticks_timestamps = []
        for i in estim_param.tint_eval_dict.keys():
            self.axis_ticks_timestamps.append(estim_param.tint_eval_dict[i].split()[1].replace(':00', ''))

        #estim_param.print_incr_runtime()
        print "(completed)"
        
    def getFlowSubVector(self, estim_param, network, edges_sub_dict):
        subvec = np.zeros(len(estim_param.tint_eval_dict) * len(edges_sub_dict))
        subvec_std = np.zeros(len(estim_param.tint_eval_dict) * len(edges_sub_dict))
    
        for edge_key in edges_sub_dict.keys():
            edge = edges_sub_dict[edge_key]
            edge_id = network.edges_dict_rev[edge]
    
            for tint in estim_param.tint_eval_dict:
                subvec[tint * len(edges_sub_dict) + edge_key] = self.flow_mean[tint * len(network.edges_dict) + edge_id]
                subvec_std[tint * len(edges_sub_dict) + edge_key] = self.flow_std[tint * len(network.edges_dict) + edge_id]
    
        return subvec, subvec_std
        

def mean_std(dist):
    dist_mean = np.mean(dist, axis=1)
    dist_std = np.std(dist, axis=1)
        
    return dist_mean, dist_std
