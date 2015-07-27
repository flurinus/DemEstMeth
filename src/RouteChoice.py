'''
Created on Jul 3, 2014

@author: flurin
'''
import numpy as np
import datetime
import pandas as pd


def sales_data_GLS(estim_param, network, dest_flows):
    '''
    Sales over time
    '''
    M_sales = np.zeros((2 * len(network.shops_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
    v_sales = np.zeros(2 * len(network.shops_dict))

    for shop_key in network.shops_dict.keys():
        for route_key in network.routes_from_centroid[network.centroids_dict_rev[network.shops_dict[shop_key]]]:
            for tint_key in estim_param.tint_dict.keys():
                M_sales[shop_key, tint_key * len(network.routes_dict) + route_key] = 1
    
        for route_key in network.routes_to_centroid[network.centroids_dict_rev[network.shops_dict[shop_key]]]:
            for tint_key in estim_param.tint_dict.keys():
                M_sales[shop_key + len(network.shops_dict), tint_key * len(network.routes_dict) + route_key] = 1
                
        v_sales[shop_key] = dest_flows[network.centroids_dict_rev[network.shops_dict[shop_key]]]
        v_sales[shop_key + len(network.shops_dict)] = v_sales[shop_key]

    return M_sales, v_sales

    
def platform_centroid_dep_flow(estim_param, network, dest_flows):
    '''
    Departures from platforms over time
    '''
    M_depflow = np.zeros((len(network.centroids_p_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
    v_depflow = np.zeros(len(network.centroids_p_dict))

    for centroid_p_key in network.centroids_p_dict.keys():
        for route_key in network.routes_to_centroid[network.centroids_dict_rev[network.centroids_p_dict[centroid_p_key]]]:
            for tint_key in estim_param.tint_dict.keys():
                M_depflow[centroid_p_key, tint_key * len(network.routes_dict) + route_key] = 1
                
        v_depflow[centroid_p_key] = dest_flows[network.centroids_dict_rev[network.centroids_p_dict[centroid_p_key]]]

    return M_depflow, v_depflow

    
def agg_cum_route_split_GLS(estim_param, network, data, dest_flows):
    '''
    Platform/non-platform aggragated cumulative route split fractions
    '''
    M_agg_rsplits = np.zeros((len(network.centroids_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
    v_agg_rsplits = np.zeros(len(network.centroids_dict))
        
    for centroid_key in network.centroids_dict:
        orig_type_key = centroid_type_RCh_key(network, centroid_key)

        split_to_np = ped_type_fraction(network, data, orig_type_key, network.centroid_types_RCh_dict_rev['non-platform'])

        for route_key in network.routes_from_centroid[centroid_key]:
            dest_key = network.route_dest_dict[route_key]
            dest_type_key = centroid_type_RCh_key(network, dest_key)

            if network.centroid_types_RCh_dict[dest_type_key] == 'non-platform':
                coeff = 1 - split_to_np
            else:
                coeff = - split_to_np
                
            for tint_key in estim_param.tint_dict.keys():
                M_agg_rsplits[centroid_key, route_key + tint_key * len(network.routes_dict)] = coeff
                
    return M_agg_rsplits, v_agg_rsplits

    
def route_split_GLS(estim_param, network, data, dest_flows):
    '''
    Quasi-static route split fractions
    '''
    M_rsplits = np.zeros((len(network.routes_dict) * len(estim_param.tint_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
    v_rsplits = np.zeros(len(network.routes_dict) * len(estim_param.tint_dict))
    
    route_splits = compute_route_splits(network, data, dest_flows)
    
    for route_key in network.routes_dict.keys():
        for route_prime_key in network.routes_from_centroid[network.route_orig_dict[route_key]]:
            if route_prime_key == route_key:
                coeff = 1 - route_splits[route_prime_key]
            else:
                coeff = - route_splits[route_prime_key]
                
            for tint_key in estim_param.tint_dict.keys():
                M_rsplits[route_key + tint_key * len(network.routes_dict), route_prime_key + tint_key * len(network.routes_dict)] = coeff
                
    return M_rsplits, v_rsplits

    
def compute_route_splits(network, data, dest_flows):
    '''
    Returns vector of route split fractions ordered by routes_dict
    '''
    orig_flow_by_dest_type = origin_flow_by_destination_type(network, dest_flows)  # Matrix centroid x type
    
    route_splits = np.zeros(len(network.routes_dict))
        
    for route_key in network.routes_dict.keys():
        orig_key = network.route_orig_dict[route_key]
        dest_key = network.route_dest_dict[route_key]
        orig_type_key = centroid_type_RCh_key(network, orig_key)
        dest_type_key = centroid_type_RCh_key(network, dest_key)
        
        route_splits[route_key] = ped_type_fraction(network, data, orig_type_key, dest_type_key) * dest_flows[dest_key] / orig_flow_by_dest_type[orig_key, dest_type_key]
        
    return route_splits


def origin_flow_by_destination_type(network, dest_flows):
    '''
    OD flows creation, origin part
    '''
    orig_flow_by_dest_type = np.zeros((len(network.centroids_dict), len(network.centroid_types_RCh_dict)))
    
    for route_key in network.routes_dict.keys():
        orig_key = network.route_orig_dict[route_key]
        dest_key = network.route_dest_dict[route_key]
        dest_type_key = centroid_type_RCh_key(network, dest_key)

        orig_flow_by_dest_type[orig_key, dest_type_key] += dest_flows[dest_key]
    
    return orig_flow_by_dest_type

    
# returns beta_{orig_type,dest_type}
def ped_type_fraction(network, data, orig_type_key, dest_type_key):
    '''
    Returns pedestrian fractions
    '''
    if network.centroid_types_RCh_dict[orig_type_key] == 'platform':
        if network.centroid_types_RCh_dict[dest_type_key] == 'non-platform':
            return data.beta_p2np
        else:
            return 1 - data.beta_p2np
    else:
        if network.centroid_types_RCh_dict[dest_type_key] == 'platform':
            return data.beta_np2p
        else:
            return 1 - data.beta_np2p
        

def generate_dest_flows(estim_param, network, data):
    '''
    Rxeturns cumulative destination flows at centroids over time ordered by centroids_dict
    '''
    dest_flows = np.zeros(len(network.centroids_dict))
    
    cum_flow_to_platforms_dict = generate_platform_dest_flow(estim_param, network)
    
    for centroid_key in network.centroids_dict.keys():
        centr_type = centroid_type_detail_value(network, centroid_key)
        
        if centr_type == 'entrance':
                cum_flow = 0
                exit_link = network.exit_centroids_exit_edges[network.centroids_dict[centroid_key]]
                
                ASE_LS = data.ASEdataPreprocessing(network)
                for t_key in estim_param.tint_dict.keys():
                    t_int = estim_param.tint_dict[t_key]
                    t_int = datetime.datetime.strptime(t_int, estim_param.date_format)
                    cum_flow += ASE_LS.ix[t_int, exit_link]
                
                dest_flows[centroid_key] = cum_flow

        elif centr_type == 'platform':
            dest_flows[centroid_key] = cum_flow_to_platforms_dict[network.centroid_platform_dict[network.centroids_dict[centroid_key]]] * data.centroid_platform_dest_flow_fractions[network.centroids_dict[centroid_key]]
            
        else:
            dest_flows[centroid_key] = data.sales_per_minute[network.centroids_dict[centroid_key]] * estim_param.num_tint
        
    return dest_flows

    
def centroid_type_detail_value(network, centroid_key):
    '''
    Mapping helper function
    '''
    if network.centroids_dict[centroid_key] in network.centroids_p_dict.values():
        return 'platform'
    elif network.centroids_dict[centroid_key] in network.centroids_ee_dict.values():
        return 'entrance'
    else:
        return 'shop'

        
def centroid_type_RCh_key(network, centroid_key):
    '''
    Mapping helper function
    '''
    if network.centroids_dict[centroid_key] in network.centroids_p_dict.values():
        return network.centroid_types_RCh_dict_rev['platform']
    else:
        return network.centroid_types_RCh_dict_rev['non-platform']

        
def generate_platform_dest_flow(estim_param, network):
    '''
    Creates the flow heading towrads a platform
    '''
    FQ = pd.read_csv(estim_param.path_FQ)

    cum_flow_to_platforms = np.zeros(len(network.platforms_dict))
    
    for i in FQ.index.values:
        if isinstance(FQ.ix[i, 't_dep_sched'], str):
            td_date = datetime.datetime.strptime(estim_param.date + " " + FQ.ix[i, 't_dep_sched'] + ":00", estim_param.date_format)
            if estim_param.start_date_t < td_date and td_date <= estim_param.end_date_t:
                platform = network.track_platform_dict[FQ.ix[i, 'track_sched']]
                
                if np.isnan(FQ.ix[i, 'dep_HOP']):
                    boarding = FQ.ix[i, 'dep_FRASY']
                else:
                    boarding = FQ.ix[i, 'dep_HOP']
                                    
                cum_flow_to_platforms[network.platforms_dict_rev[platform]] += boarding
                
    cum_flow_to_platforms_dict = {}
    for platform_key in network.platforms_dict.keys():
        cum_flow_to_platforms_dict[network.platforms_dict[platform_key]] = cum_flow_to_platforms[platform_key]
        
    return cum_flow_to_platforms_dict
