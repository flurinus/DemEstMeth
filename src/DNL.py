'''
Created on Mar 26, 2014

@author: flurin
'''
import scipy as sp
import numpy as np
import scipy.integrate as integ
import scipy.stats as stats
import networkx as nx

#Multiprocessing
import multiprocessing
from functools import partial


def compute_assg_mat(network, estim_param):
    '''
    computes the full link assignment matrix (B in paper), for a given network and a given
    speed distribution (Weidmann, 1993) passed in estim_param
    '''
    delta_t = estim_param.delta_t
    tint_dict = estim_param.tint_dict
    max_TT_int = estim_param.max_TT_int
    
    routes_dict = network.routes_dict
    edges_dict = network.edges_dict
    
    # dist_route_edge_{r,e}: r and e sorted by the corresponding dictionaries
    dist_route_edge = sp.zeros((len(routes_dict), len(edges_dict)))
    for r in routes_dict.keys():
        for e in edges_dict.keys():
            if '(' + str(edges_dict[e][0]) + ',' + str(edges_dict[e][1]) + ')' in ['(' + str(routes_dict[r][i - 1]) + ',' + str(routes_dict[r][i]) + ')' for i in range(1, len(routes_dict[r]))]:
                dist_route_edge[r][e] = path_length_to_node(network.G, routes_dict[r], edges_dict[e][0])
            else:
                dist_route_edge[r][e] = -1
        
    if __name__ == '__main__' or estim_param.good_op_sys is True:
        po = multiprocessing.Pool(estim_param.parallel_threads)
    
    #We initialize the matrices with zeros
    linkFlowAss = sp.zeros((len(edges_dict) * len(tint_dict), len(routes_dict) * len(tint_dict)))
    D = sp.zeros((len(edges_dict) * max_TT_int, len(routes_dict)))  # Time-invariant version of linkFlowAss

    for r in routes_dict.keys():
        for h_minus_t in range(max_TT_int):
            for e in edges_dict.keys():
                d_e_r = dist_route_edge[r][e]
                t = 0
                h = h_minus_t
                #cf: http://glench.com/articles/python-async-callback.html
                new_callback = partial(multiproc_callback, matrix=D, h_minus_t=h_minus_t, r=r, obj=e, num_obj=len(edges_dict))

                #compute the matrix entry for the given edge, route, time elapsed between the two time intervals.
                if __name__ == '__main__' or estim_param.good_op_sys is True:
                    po.apply_async(compute_a_gaussian, args=(h, t, d_e_r, delta_t, estim_param), callback=new_callback)
                    
    if __name__ == '__main__' or estim_param.good_op_sys is True:
        po.close()
        po.join()  # Execution stops here until all computations are completed

    #store results into the complete assignment matrix
    for r in routes_dict.keys():
        for t in tint_dict.keys():
            for e in edges_dict.keys():
                for h in range(t, min(t + max_TT_int, len(tint_dict.keys()))):  # No arrival prior to departing time int
                    linkFlowAss[h * len(edges_dict) + e, t * len(routes_dict) + r] = D[(h - t) * len(edges_dict) + e, r]
    return linkFlowAss


def build_flow_assg_mat(linkFlowAss, network, estim_param, sensor_edge_dict):
    '''
    From the full flow assigment matrix, we compute the
    reduced flow assigment matrix based on the existence of data.
    '''
    linkFlowAssReduced = sp.zeros((len(sensor_edge_dict) * len(estim_param.tint_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
  
    for tint in estim_param.tint_dict.keys():
        for sensid in sensor_edge_dict.keys():
            linkFlowAssReduced[tint * len(sensor_edge_dict) + sensid, :] = linkFlowAss[tint * len(network.edges_dict) + network.edges_dict_rev[sensor_edge_dict[sensid]], :]
    
    return linkFlowAssReduced


def build_ODflow_assg_mat(linkFlowAss, network, estim_param, sensor_routes_dict):
    '''
    From the full flow assigment matrix, we compute the
    reduced subroute flow assigment matrix based on the existence of data.
    '''
    subRouteFlowAssReduced = sp.zeros((len(sensor_routes_dict) * len(estim_param.tint_dict), len(network.routes_dict) * len(estim_param.tint_dict)))
       
    for idx_sensor_route in sensor_routes_dict.keys():
        for idx_route in network.routes_dict.keys():
            if str(sensor_routes_dict[idx_sensor_route])[1:-1] in str(network.routes_dict[idx_route]):
                
                sensor_edge = (sensor_routes_dict[idx_sensor_route][0], sensor_routes_dict[idx_sensor_route][1])  # First edge of sensor route
                sensor_edge_idx = network.edges_dict_rev[sensor_edge]
                
                #print "route", idx_route, network.routes_dict[idx_route], "contains subroute", idx_sensor_route, sensor_routes_dict[idx_sensor_route], "start_edge", sensor_edge_idx, sensor_edge
                                
                for dep_tint in estim_param.tint_dict.keys():
                    for arr_tint in estim_param.tint_dict.keys():
                        subRouteFlowAssReduced[arr_tint * len(sensor_routes_dict) + idx_sensor_route, dep_tint * len(network.routes_dict) + idx_route] = linkFlowAss[arr_tint * len(network.edges_dict) + sensor_edge_idx, dep_tint * len(network.routes_dict) + idx_route]

    return subRouteFlowAssReduced

    
def compute_a_gaussian(h, t, d_e_r, delta_t, estim_param):
    '''
    Computes an element of the flow assignment matrix, when the speed distribution is normal
    '''

    if h < t:  # We cannot arrive before leaving
        return 0

    if t == h and d_e_r == 0:
        return 1

    if d_e_r != -1:  # We do not compute if the edge is not on the route
        h_minus = h * delta_t
        h_plus = (h + 1) * delta_t
        t_minus = t * delta_t
        t_plus = (t + 1) * delta_t

        res = integ.quad(Gaussian_aux, t_minus, t_plus, args=(d_e_r, h_minus, h_plus, estim_param))[0]
        
        return (1.0 / delta_t) * res

    else:
        return 0


def Gaussian_aux(d, d_e_r, h_minus, h_plus, estim_param):
    """ This is a simple auxiliary function that will be integrated when computing each element of the assignment matrix. Returns CDF(dist/(h- - t)) - CDF(dist/(h+ - t)) """

    vf_mu = estim_param.vel_mean
    vf_sigma = estim_param.vel_stdev

    res = stats.norm.cdf(d_e_r / (h_minus - d), vf_mu, vf_sigma) - stats.norm.cdf(d_e_r / (h_plus - d), vf_mu, vf_sigma)
    
    # This happens when h_minus - t < 0 (i.e. h is t, see report)
    if res < 0:
        res = 1 + res

    return res
# occupation


def compute_assg_mat_accumulation(network, estim_param):
    '''
    Creates the accumulation assignment matrix
    '''

    tint_dict = estim_param.tint_dict
    max_TT_int = estim_param.max_TT_int
    
    routes_dict = network.routes_dict
    areas_dict = network.areas_dict

    S = sp.zeros((len(areas_dict) * len(tint_dict), len(routes_dict) * len(tint_dict)))
    R = sp.zeros((len(areas_dict) * max_TT_int, len(routes_dict)))  # Time-invariant version of S
      
    #compute entrance and exit distance for all pairs of routes and areas
    dist_routes_areas = distance_route_area(network)
    
    if __name__ == '__main__' or estim_param.good_op_sys is True:
        po = multiprocessing.Pool(estim_param.parallel_threads)

    # For each route, we know its distances to the PI, we compute the corresponding matrix entry
    for area in areas_dict.keys():
        for route in routes_dict.keys():
            d_in = dist_routes_areas[route][area][0]
            d_out = dist_routes_areas[route][area][1]
            if d_in > -1 and d_out > -1:
                for h_minus_t in range(max_TT_int):
                    new_callback = partial(multiproc_callback, matrix=R, h_minus_t=h_minus_t, r=route, obj=area, num_obj=len(areas_dict))
                    if __name__ == '__main__' or estim_param.good_op_sys is True:
                        po.apply_async(accumulation_aux_Gaussian, args=(d_in, d_out, h_minus_t, estim_param), callback=new_callback)
    if __name__ == '__main__' or estim_param.good_op_sys is True:
        po.close()  # close the pool
        po.join()  # make sure that we got all the results
    for a in areas_dict.keys():
        for t in tint_dict.keys():
            for r in routes_dict.keys():
                for h in range(t, min(t + max_TT_int, len(tint_dict))):  # No arrival prior to departing time int
                    S[h * len(areas_dict) + a, t * len(routes_dict) + r] = R[(h - t) * len(areas_dict) + a, r]
    return S


def distance_route_area(network):
    '''
    For a given route and area, returns the distance to
    the entrance and exit of the area from the start of a route.
    But this method calculates these distances for all routes and areas.
    '''
    routes_dict = network.routes_dict
    areas_dict = network.areas_dict

    distances_route_area = sp.zeros((len(routes_dict), len(areas_dict), 2))

    for area in areas_dict.keys():
        for route in routes_dict.keys():
            min_dist = -1
            max_dist = -1
            for node in routes_dict[route]:
                if node in areas_dict[area]:
                    min_dist = path_length_to_node(network.G, routes_dict[route], node)
                    break
                
            reversed_route = routes_dict[route][::-1]
            for node in reversed_route:
                if node in areas_dict[area]:
                    max_dist = path_length_to_node(network.G, routes_dict[route], node)
                    break
            
            distances_route_area[route][area][:] = [min_dist, max_dist]

    return distances_route_area

    
def accumulation_aux_Gaussian(d_in, d_out, h_minus_t, estim_param):
    '''
    Helper for the accumulation_Gaussian method
    '''
    return 1.0 / (estim_param.delta_t ** 2) * integ.quad(accumulation_Gaussian, 0, sp.inf, args=(d_in, d_out, h_minus_t, estim_param))[0]

    
def accumulation_Gaussian(v, dist_in, dist_out, delta_tint, estim_param):
    '''
    Performs the integration of the pedestrian accumulation
    '''
    gener_tint_minus = delta_tint * estim_param.delta_t
    gener_tint_plus = (delta_tint + 1) * estim_param.delta_t
    return stats.norm.pdf(v, estim_param.vel_mean, estim_param.vel_stdev) * integ.quad(residence_time_aux, 0, estim_param.delta_t, args=(v, dist_in, dist_out, gener_tint_minus, gener_tint_plus))[0]

    
def residence_time_aux(t, v, dist_in, dist_out, tint_minus, tint_plus):
    '''
    Helper for residence_time
    '''
    t_start = tint_minus - t
    t_end = tint_plus - t
    return residence_time(v, dist_in, dist_out, t_start, t_end)

    
def residence_time(v, dist_in, dist_out, t_start, t_end):
    '''
    Calculates the residence time based on the times and distances
    '''
    t_in = dist_in / v
    t_out = dist_out / v

    if t_start <= t_in <= t_end <= t_out:
        return t_end - t_in
    elif t_in <= t_start <= t_out <= t_end:
        return t_out - t_start
    elif t_in <= t_start <= t_end <= t_out:
        return t_end - t_start
    elif t_start <= t_in <= t_out <= t_end:
        return t_out - t_in
    else:
        return 0

        
def multiproc_callback(res, matrix, h_minus_t, r, obj, num_obj):
    '''
    This function is called when a computation is finished, and stores the result in the given matrix
    '''
    matrix[h_minus_t * num_obj + obj, r] = res

    
def path_length_to_node(G, route, node):
    '''
    Returns the length of the route up to a given node.
    This function works as the paths are the "shortest paths"
    '''
    return nx.shortest_path_length(G, source=route[0], target=node, weight='length')

    
def apply_assg(assg_mat, route_demand):
    '''
    Dot product between both arguments
    '''
    return sp.dot(np.asarray(assg_mat), route_demand)
