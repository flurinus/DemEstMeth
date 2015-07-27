'''
Created on Mar 25, 2014

@author: flurin
'''
import numpy as np
import DNL


class Evaluation:
    '''
    Evaluates the solution found from the previous calculations.
    Basically procssing the results ready for post-processing (visualization)
    '''
    def __init__(self, estim_param, network, assignment, data, prior, estimate):
        '''
        Constructor
        '''
        print "Evaluating demand",

        # demand
        self.demand_mean, self.demand_std, self.demand_dist = self.monteCarloDemand(estim_param, network, estimate)
        self.demand_agg_time_mean, self.demand_agg_time_std = aggreg_time(self.demand_dist, estim_param.tint_eval_dict.keys(), network.routes_dict)
        self.demand_agg_space_mean, self.demand_agg_space_std = aggreg_space(self.demand_dist, estim_param.tint_eval_dict.keys(), network.routes_dict)
        
        # edge flows
        self.flow_mean, self.flow_std, self.flow_dist = self.monteCarloIndicator(estim_param, estimate, network.edges_dict, assignment.linkFlowAssgn)
        
        self.flow_agg_time_mean, self.flow_agg_time_std = aggreg_time(self.flow_dist, estim_param.tint_eval_dict.keys(), network.edges_dict)
        
        # area occupation
        self.occ_est_mean, self.occ_est_std, self.occ_est_dist = self.monteCarloIndicator(estim_param, estimate, network.areas_dict, assignment.acc)
        self.occ_est_agg_space_mean, self.occ_est_agg_space_std = aggreg_space(self.occ_est_dist, estim_param.tint_eval_dict.keys(), network.areas_dict)
        self.occ_meas_mean = self.eval_part_of_vector(estim_param, data.n_hat, network.areas_dict)
        self.occ_meas_std = None
        self.occ_meas_agg_space_mean, self.occ_meas_agg_space_std = aggreg_space(self.occ_meas_mean, estim_param.tint_eval_dict.keys(), network.areas_dict)

        # estimated subroute demand_mean
        self.sub_dem_est_mean, self.sub_dem_est_std, self.sub_dem_est_dist = self.monteCarloIndicator(estim_param, estimate, network.subroutes_VS_dict, assignment.subRouteAssgn_prime)
        self.sub_dem_est_agg_time_mean, self.sub_dem_est_agg_time_std = aggreg_time(self.sub_dem_est_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        self.sub_dem_est_agg_space_mean, self.sub_dem_est_agg_space_std = aggreg_space(self.sub_dem_est_dist, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        
        # measured subroute demand_mean
        self.sub_dem_meas_mean = self.eval_part_of_vector(estim_param, data.g_hat, network.subroutes_VS_dict)
        self.sub_dem_meas_std = None
        self.sub_dem_meas_agg_time_mean, self.sub_dem_meas_agg_time_std = aggreg_time(self.sub_dem_meas_mean, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        self.sub_dem_meas_agg_space_mean, self.sub_dem_meas_agg_space_std = aggreg_space(self.sub_dem_meas_mean, estim_param.tint_eval_dict.keys(), network.subroutes_VS_dict)
        
        # ASE flows
        self.ASE_flow_est_mean, self.ASE_flow_est_std = self.getFlowSubVector(estim_param, network, network.edges_ASE_dict)
        self.ASE_flow_meas_mean = self.eval_part_of_vector(estim_param, data.f_hat, network.edges_ASE_dict)
        self.ASE_flow_meas_std = None
        
        # TINF flows
        self.TINF_flow_est_mean, self.TINF_flow_est_std = self.getFlowSubVector(estim_param, network, network.edges_TINF_dict)
        self.TINF_flow_prior_dist = self.eval_part_of_matrix(estim_param, prior.tinf_distr, network.edges_TINF_dict)
        self.TINF_flow_prior_mean = self.eval_part_of_vector(estim_param, prior.tinf_mean, network.edges_TINF_dict)
        self.TINF_flow_prior_std = self.eval_part_of_vector(estim_param, prior.tinf_std, network.edges_TINF_dict)

        # VS flows: returns estimated and measured in- and outflows
        self.compute_VS_flows(estim_param, network, data)
        
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
            
        estim_param.print_incr_runtime()
                
    def monteCarloDemand(self, estim_param, network, estimate):
        '''
        Calculates some statistics from the MC simulation runs
        '''
        start_index = estim_param.tint_eval_start * len(network.routes_dict)
        end_index = (estim_param.tint_eval_start + len(estim_param.tint_eval_dict)) * len(network.routes_dict)
        
        if estim_param.MC_iterations > 1:
            demand_eval = np.mean(estimate.demand_distr[start_index:end_index, :], axis=1)
            demand_eval_std = np.std(estimate.demand_distr[start_index:end_index, :], axis=1)
            dem_dist = estimate.demand_distr[start_index:end_index, :]
        else:
            demand_eval = estimate.demand_distr[start_index:end_index]
            demand_eval_std = None
            dem_dist = demand_eval
            
        return demand_eval, demand_eval_std, dem_dist
    
    def monteCarloIndicator(self, estim_param, estimate, space_dict, assg_map):
        '''
        Calculates indicators from the MC runs
        '''
        start_index = estim_param.tint_eval_start * len(space_dict)
        end_index = (estim_param.tint_eval_start + len(estim_param.tint_eval_dict)) * len(space_dict)
        
        if estim_param.MC_iterations > 1:
            IndDist = np.zeros((len(estim_param.tint_dict) * len(space_dict), estim_param.MC_iterations))
            for iteration in range(estim_param.MC_iterations):
                IndDist[:, iteration] = np.asarray(DNL.apply_assg(assg_map, estimate.demand_distr[:, iteration]))
        
            ind_mean = np.mean(IndDist[start_index:end_index, :], axis=1)
            ind_std = np.std(IndDist[start_index:end_index, :], axis=1)
            ind_dist = IndDist[start_index:end_index, :]
                       
        else:
            ind_mean = np.asarray(DNL.apply_assg(np.asarray(assg_map), estimate.demand_distr))[start_index:end_index]
            ind_std = None
            ind_dist = ind_mean

        return ind_mean, ind_std, ind_dist

    def eval_part_of_vector(self, estim_param, data_vec, space_vector):
        '''
        Subsets the vector to the adequate size based on the evaluation period
        '''
        return np.asarray(data_vec[estim_param.tint_eval_start * len(space_vector):(estim_param.tint_eval_start + len(estim_param.tint_eval_dict)) * len(space_vector)])
    
    def eval_part_of_matrix(self, estim_param, data_mat, space_vector):
        '''
        Subsets the matrix based on the evalutaion period
        '''
        return np.asarray(data_mat[estim_param.tint_eval_start * len(space_vector):(estim_param.tint_eval_start + len(estim_param.tint_eval_dict)) * len(space_vector), :])
    
    def getFlowSubVector(self, estim_param, network, edges_sub_dict):
        '''
        Gets the flows on the reduced vector
        '''
        subvec = np.zeros(len(estim_param.tint_eval_dict) * len(edges_sub_dict))
        
        if not self.flow_std is None:
            subvec_std = np.zeros(len(estim_param.tint_eval_dict) * len(edges_sub_dict))
        else:
            subvec_std = None
    
        for edge_key in edges_sub_dict.keys():
            edge = edges_sub_dict[edge_key]
            edge_id = network.edges_dict_rev[edge]
    
            for tint in estim_param.tint_eval_dict:
                subvec[tint * len(edges_sub_dict) + edge_key] = self.flow_mean[tint * len(network.edges_dict) + edge_id]
                if not self.flow_std is None:
                    subvec_std[tint * len(edges_sub_dict) + edge_key] = self.flow_std[tint * len(network.edges_dict) + edge_id]
    
        return subvec, subvec_std
        
    def compute_VS_flows(self, estim_param, network, data):
        '''
        Computes the flows from the simuation, to compare with the VS flows
        '''
        self.VS_inflow_meas_mean = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
        self.VS_outflow_approx_meas_mean = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
        
        self.VS_inflow_meas_std = None
        self.VS_outflow_approx_meas_std = None
    
        self.VS_inflow_est_mean = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
        self.VS_outflow_est_mean = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
        
        if self.flow_std is None:
            self.VS_inflow_est_std = None
            self.VS_outflow_est_std = None
        else:
            self.VS_inflow_est_std = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
            self.VS_outflow_est_std = np.zeros(len(estim_param.tint_eval_dict) * len(network.VS_nodes_dict))
 
        for node_key in network.VS_nodes_dict.keys():
    
            inflow_edge = network.VS_inflow_edges_dict[node_key]
            outflow_edge = network.VS_outflow_edges_dict[node_key]
            
            inflow_edge_id = network.edges_dict_rev[inflow_edge]
            outflow_edge_id = network.edges_dict_rev[outflow_edge]
    
            orig_flow_subroutes = network.subroutes_from_node[node_key]
            dest_flow_subroutes = network.subroutes_to_node[node_key]
            
            for t in estim_param.tint_eval_dict.keys():
                t_full = t + estim_param.tint_eval_start
    
                for subroute_id in orig_flow_subroutes:
                    self.VS_inflow_meas_mean[t * len(network.VS_nodes_dict) + node_key] += data.g_hat[t_full * len(network.subroutes_VS) + subroute_id]
    
                for subroute_id in dest_flow_subroutes:
                    self.VS_outflow_approx_meas_mean[t * len(network.VS_nodes_dict) + node_key] += data.g_hat[t_full * len(network.subroutes_VS) + subroute_id]
    
                self.VS_inflow_est_mean[t * len(network.VS_nodes_dict) + node_key] = self.flow_mean[t * len(network.edges_dict) + inflow_edge_id]
                self.VS_outflow_est_mean[t * len(network.VS_nodes_dict) + node_key] = self.flow_mean[t * len(network.edges_dict) + outflow_edge_id]
                
                if not self.flow_std is None:
                    self.VS_inflow_est_std[t * len(network.VS_nodes_dict) + node_key] = self.flow_std[t * len(network.edges_dict) + inflow_edge_id]
                    self.VS_outflow_est_std[t * len(network.VS_nodes_dict) + node_key] = self.flow_std[t * len(network.edges_dict) + outflow_edge_id]
    

def aggreg_time(dat_dist, tint_indices, space_dict):
    '''
    Aggregates data over time
    '''
    n_dat = len(space_dict)
    if len(dat_dist.shape) == 2:
        n_iter = dat_dist.shape[1]
        agg_dist = np.zeros((n_dat, n_iter))
    else:
        n_iter = 1
        agg_dist = np.zeros(n_dat)
        
    for space_key in space_dict.keys():
        for tint in tint_indices:
            agg_dist[space_key] += dat_dist[tint * n_dat + space_key]
        
    if n_iter > 1:
        ind_mean = np.mean(agg_dist, axis=1)
        ind_std = np.std(agg_dist, axis=1)
    else:
        ind_mean = agg_dist
        ind_std = None
    
    return ind_mean, ind_std

    
def aggreg_space(dat_dist, tint_indices, space_dict):
    '''
    Aggregates data over space
    '''
    n_dat = len(space_dict)
    n_tint = len(tint_indices)
    if len(dat_dist.shape) == 2:
        n_iter = dat_dist.shape[1]
        agg_dist = np.zeros((n_tint, n_iter))
    else:
        n_iter = 1
        agg_dist = np.zeros(n_tint)
        
    for i in tint_indices:
        agg_dist[i] = np.sum(dat_dist[i * n_dat:(i + 1) * n_dat], 0)
    
    if n_iter > 1:
        ind_mean = np.mean(agg_dist, axis=1)
        ind_std = np.std(agg_dist, axis=1)
    else:
        ind_mean = agg_dist
        ind_std = None
    
    return ind_mean, ind_std


def error_distances(vec_true, vec_est):
    '''
    Difference between the observation and the simualtion
    '''
    error = np.array(vec_est) - np.array(vec_true)
    
    #Computes the Root Mean Square Error of a given array.
    def rmse(v):
        v = np.array(v)
        return np.sqrt((np.mean(v ** 2)))
    
    err_dict = {}
    
    err_dict['RMSE'] = rmse(error)
    err_dict['RRMSE'] = rmse(error) / rmse(np.array(vec_true))
    err_dict['MAPE'] = np.mean(np.abs((error) / np.array(vec_true))) * 100

    return err_dict
