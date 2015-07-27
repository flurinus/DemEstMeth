'''
Created on Jul 9, 2014

@author: eduard, flurin
'''
import os
import shutil

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import datetime
from matplotlib2tikz import save as tikz_save


class PostProcessing:
    '''
    Class containing all the methods required for producing images of the results.
    The import methods are the following:
        show_network_structure: this creates the static network
        show_dynamic_OD_matrix: the OD matrix is plotted using densities
        show_dynamic_flow_map: the network at each time step with the accumulation of pedestrians
        create_circos: creates the data in Circos plot format
    '''
    
    def __init__(self, estim_param, network, evaluate, estpar_ref=None, eval_ref=None):
        '''
        Constructor
        '''
        print "Postprocessing",
        
        generate_output_dir(estim_param)
        
        ## show network structure
        if estim_param.show_network_structure:
            show_network_structure(estim_param, network)
            
        # show network flow map
        if estim_param.show_dynamic_flow_map:
            show_dynamic_flow_map(estim_param, network, evaluate)
            
        ## show dynamic OD matrix
        if estim_param.show_dynamic_OD_matrix:
            show_dynamic_OD_matrix(estim_param, network, evaluate, estpar_ref, eval_ref)
                        
        ## generate Circos plot
        if estim_param.create_circos:
            create_circos(estim_param, network, evaluate)
            
        if estim_param.show_subroute_demand:
            show_subroute_demand(estim_param, evaluate, estpar_ref, eval_ref)
            
        if estim_param.show_total_demand:
            show_total_demand(estim_param, evaluate, estpar_ref, eval_ref)
            
        if estim_param.show_occupation:
            show_occupation(estim_param, evaluate, estpar_ref, eval_ref)
            
        if estim_param.compare_ASE_VS:
            self.error_ASE_VS = compare_ASE_VS(estim_param, network, evaluate, estpar_ref, eval_ref)
         
        if estim_param.compare_TINF_VS:
            self.error_TINF_VS = compare_TINF_VS(estim_param, network, evaluate, estpar_ref, eval_ref)
             
        if estim_param.show_TINF:
            show_TINF(estim_param, network, evaluate, estpar_ref, eval_ref)
             
        if estim_param.show_ASE:
            show_ASE(estim_param, network, evaluate, estpar_ref, eval_ref)
             
        if estim_param.show_VS_flows:
            show_VS_flows(estim_param, network, evaluate, estpar_ref, eval_ref)
        
        if estim_param.show_scatter_plots:
            show_scatter_plots(estim_param, evaluate, estpar_ref, eval_ref)
            
        plt.close("all")
        
        write_protocol(estim_param, network, evaluate, estpar_ref, eval_ref, self)
        
        estim_param.print_incr_runtime()
        
        print "Simulation done",
        estim_param.print_incr_runtime(time_elaps='full')
        print "***** \n\n"

        ## plots the graph of the network.


def show_network_structure(estim_param, network):
    '''
    Creates a figure with the network drawn.
    '''
    edge_labels2 = {}
    for key, value in network.edge_labels_duplicated.items():
        if key not in network.edge_labels_simple.keys():
            edge_labels2[key] = value
    
    fig = plt.figure(figsize=(16.0, 10))
    fig.add_axes((0, 0, 1, 1))
                      
    nx.draw_networkx_nodes(network.G, network.positions, network.centroids, node_color='#FFFF00', node_size=450)
    nx.draw_networkx_nodes(network.G, network.positions, network.ASE_measurement_nodes, node_color='r', node_size=150)
    nx.draw_networkx_nodes(network.G, network.positions, network.VS_measurement_nodes, node_color='y', node_size=150)
    nx.draw_networkx_nodes(network.G, network.positions, network.other_nodes, node_color='c', node_size=150)

    nx.draw_networkx_edges(network.G, network.positions, arrows=False)
    nx.draw_networkx_labels(network.G, network.positions, network.centroids_labels)
    nx.draw_networkx_labels(network.G, network.positions, network.not_centroids_labels, font_size=8, font_weight='bold')

    nx.draw_networkx_edge_labels(network.G, network.positions, edge_labels=network.edge_labels_simple, font_size=8)
    nx.draw_networkx_edge_labels(network.G, network.positions, edge_labels=edge_labels2, font_size=8)

    plt.savefig(estim_param.path_plots + "network_composition" + estim_param.file_format, dpi=400)
        
    if estim_param.plot_show:
        plt.show()


## Plots dynamic visualization of edge and origin flows
def show_dynamic_flow_map(estim_param, network, evaluate):
    '''
    Creates the sequence of images for each time step of the accumulation
    of pedestrians on each edge of the network.
    '''
    fig = plt.figure()
    fig.add_subplot(111)
    
    def update(t):
        fig.clf()
        
        plt.axis('off')
        if t >= 0:
            fig.suptitle(estim_param.case_study_name + ": " + str(estim_param.tint_eval_dict[t]) + ' -- ' + str(datetime.datetime.time(estim_param.start_date_t + datetime.timedelta(seconds=estim_param.delta_t * (t + 1 + estim_param.tint_eval_start))).strftime(estim_param.time_format)))
        elif t == -1:
            fig.suptitle(estim_param.case_study_name + ": " + str(estim_param.tint_eval_dict[0]) + ' -- ' + str(datetime.datetime.time(estim_param.start_date_t + datetime.timedelta(seconds=estim_param.delta_t * (len(estim_param.tint_eval_dict) + estim_param.tint_eval_start))).strftime(estim_param.time_format)))

        nodes_size = np.zeros((len(network.centroids),))
        edges_size = np.zeros((len(network.edges),))

        if t >= 0:
            #We assign a the demand to each node giving to it a different size depending on the size of this demand.
            for idx, route in network.routes_dict.iteritems():
                nodes_size[network.centroids_dict_rev[route[0]]] += evaluate.demand_mean[t * network.number_of_routes + idx]
                
            for e in network.edges_dict.keys():
                edges_size[e] += evaluate.flow_mean[t * len(network.edges) + e] + evaluate.flow_mean[t * len(network.edges) + network.edges_dict_rev[network.edges_reverse_dict[network.edges_dict[e]]]]
        elif t == -1:
            for idx, route in network.routes_dict.iteritems():
                # avg origin flow per minute
                nodes_size[network.centroids_dict_rev[route[0]]] += evaluate.demand_agg_time_mean[idx] / len(estim_param.tint_eval_dict)
                
            for e in network.edges_dict.keys():
                # avg edge flow per minute
                edges_size[e] += (evaluate.flow_agg_time_mean[e] + evaluate.flow_agg_time_mean[network.edges_dict_rev[network.edges_reverse_dict[network.edges_dict[e]]]]) / len(estim_param.tint_eval_dict)
            
        nx.draw_networkx_nodes(network.G, pos=network.positions, node_color='0.75', node_size=20, linewidths=0.5)
        nx.draw_networkx_nodes(network.G, pos=network.positions, node_color='#FFFFFF', nodelist=network.centroids, labels=network.node_labels, node_size=(nodes_size + 5) * 5)
        
        #generate gray border of each edge
        nx.draw_networkx_edges(network.G, pos=network.positions, edge_color='0.75', width=6.5, arrows=False)
        
        #generate colored edges
        a = nx.draw_networkx_edges(network.G, pos=network.positions, edge_color=edges_size, edge_vmin=0, edge_vmax=100, width=5, edge_cmap=plt.cm.get_cmap('OrRd'), arrows=False)
   
        cbar = fig.colorbar(a, ticks=[0, 25, 50, 75, 100], orientation='horizontal')
        cbar.ax.set_xticklabels(['0', '25', '50', '75', '>100 ped/min'])  # horizontal colorbar

        ax = fig.gca()
        newax = fig.add_axes(ax.get_position(), frameon=False)
        plt.ylim(-100, 0)
        plt.xlim(-50, 0)
        newax.scatter(-20, -94, s=(100 + 5) * 5, c='w')
        newax.scatter(-35, -94, s=(10 + 5) * 5, c='w')
        plt.axis('off')
        plt.annotate('100 ped/min', xy=(0.64, 0.045), xycoords='axes fraction')
        plt.annotate('10 ped/min', xy=(0.315, 0.045), xycoords='axes fraction')
        
        if t < 0:
            counter = 'agg_time'
        else:
            counter = ('0' + str(t))[-2:]

        plt.savefig(estim_param.path_plots + "flow_map-" + counter + estim_param.file_format, dpi=400)
    
    if estim_param.plot_show:
        anim = animation.FuncAnimation(fig, update, interval=500, frames=len(estim_param.tint_eval_dict), repeat=False, blit=False)
        
        plt.show()
    else:
        update(-1)  # generates time-aggregated flow map
        for i in range(len(estim_param.tint_eval_dict)):
            update(i)


def show_dynamic_OD_matrix(estim_param, network, evaluate, estpar_ref, eval_ref):
    """ Method creating the OD matrix visualization. """
    fig = plt.figure(1)

    def plot_cur_ODmap(t, flag=None):

        fig.clf()
        plt.axis('off')
        
        if t >= 0:
            time_span = str(estim_param.tint_eval_dict[t]) + ' -- ' + str(datetime.datetime.time(estim_param.start_date_t + datetime.timedelta(seconds=estim_param.delta_t * (t + 1 + estim_param.tint_eval_start))).strftime(estim_param.time_format))

        elif t == -1:
            time_span = str(estim_param.tint_eval_dict[0]) + ' -- ' + str(datetime.datetime.time(estim_param.start_date_t + datetime.timedelta(seconds=estim_param.delta_t * (len(estim_param.tint_eval_dict) + estim_param.tint_eval_start))).strftime(estim_param.time_format))

        if flag is None:
                header = estim_param.case_study_name
        elif flag == 'diff':
                header = "|" + estim_param.case_study_name + " -- " + estpar_ref.case_study_name + "|"
        elif flag == 'std':
            header = "std of " + estim_param.case_study_name
            
        fig.suptitle(header + ": " + time_span)

        table = np.zeros((len(network.structural_labels), len(network.structural_labels)))

        for idx, route in network.routes_dict.iteritems():
            if t >= 0:
                if flag is None:
                    route_demand = evaluate.demand_mean[t * network.number_of_routes + idx]
                elif flag == 'diff':
                    route_demand = evaluate.demand_mean[t * network.number_of_routes + idx] - eval_ref.demand_mean[t * network.number_of_routes + idx]
                elif flag == 'std':
                    route_demand = evaluate.demand_std[t * network.number_of_routes + idx]
            elif t == -1:
                if flag is None:
                    # avg origin flow per minute
                    route_demand = evaluate.demand_agg_time_mean[idx] / len(estim_param.tint_eval_dict)
                elif flag == 'diff':
                    route_demand = (evaluate.demand_agg_time_mean[idx] - eval_ref.demand_agg_time_mean[idx]) / len(estim_param.tint_eval_dict)
                #elif flag == 'std':
                #    route_demand = evaluate.demand_agg_time_std[idx]
            
            struct_orig = network.structural_centroids_dict[route[0]]
            struct_dest = network.structural_centroids_dict[route[-1]]
            table[struct_orig][struct_dest] += route_demand
            
        tabint = table.astype(int)

        plt.figure(1)
        if flag is None:
            cur_cmap = plt.cm.get_cmap('OrRd')
            climits = [0, 100]
        elif flag == 'diff':
            cur_cmap = plt.cm.get_cmap('RdBu_r')
            climits = [-50, 50]
        elif flag == 'std':
            cur_cmap = plt.cm.get_cmap('OrRd')
            climits = [0, 50]
            
        cax = plt.matshow(tabint, fignum=1, cmap=cur_cmap)
        plt.xlabel("Destination")
        plt.ylabel("Origin")
        plt.xticks(range(len(network.structural_labels)), network.structural_labels)
        plt.yticks(range(len(network.structural_labels)), network.structural_labels)
        plt.clim(climits)
        plt.colorbar(cax)
        
        if flag is None:
            label = ''
        elif flag == 'diff':
            label = '_diff'
        elif flag == 'std':
            label = '_std'
            
        if t < 0:
            counter = 'agg_time'
        else:
            counter = ('0' + str(t))[-2:]
            
            plt.savefig(estim_param.path_plots + "dyn_OD_map" + label + "-" + counter + estim_param.file_format)
        """ REMOVED TIKZ POSSIBILITY
        if estim_param.LaTeX:
            if flag == 'diff':
                #matplotlib2tikz has issues dealing with a negative colorbar range. Requires UTF-8 encoding
                #tikz_save(estim_param.path_plots+"dyn_OD_map" + label + "-" + counter + '.tex', show_info = False, encoding='utf-8', tex_relative_path_to_data = 'plot_output/') # does NOT work
                pass
            else:
                tikz_save(estim_param.path_plots + "dyn_OD_map" + label + "-" + counter + '.tex', show_info=False, tex_relative_path_to_data='plot_output/')
         """

    plot_cur_ODmap(-1)  # generates time-aggregated OD map
    if estpar_ref is not None:
        plot_cur_ODmap(-1, flag='diff')
    
    #if not evaluate.demand_std == None:
    #    plot_cur_ODmap(-1,flag='std') #mathematically, this doesn't make any sense
        
    for i in range(len(estim_param.tint_eval_dict)):
        plot_cur_ODmap(i)
        if estpar_ref is not None:
            plot_cur_ODmap(i, flag='diff')
            
        if not evaluate.demand_std is None:
            plot_cur_ODmap(i, flag='std')

    if estim_param.plot_show:
        anim = animation.FuncAnimation(fig, plot_cur_ODmap, interval=500, frames=len(estim_param.tint_eval_dict), repeat=False, blit=False)
        
        plt.show()
    else:
        plt.close()
    
            
def create_circos(estim_param, network, evaluate):
    '''
    Create a file with OD Demand for Circos plot (http://mkweb.bcgsc.ca/tableviewer/visualize/)
    '''
    
    odTable = np.zeros((len(network.structural_labels), len(network.structural_labels)))
    for t in estim_param.tint_eval_dict.keys():
        for idx, route in network.routes_dict.iteritems():
            odTable[network.structural_centroids_dict[route[0]]][network.structural_centroids_dict[route[-1]]] += evaluate.demand_mean[t * network.number_of_routes + idx]
    
    #We save it to file, for easy use on http://mkweb.bcgsc.ca/tableviewer/visualize/
    file_name = estim_param.path_textout + "circos.txt"
    circos_file = open(file_name, 'w')

    column_order = "-"
    
    #Two different color mappings that we can use:
    
    ## Multiple colors:
    #color_description = "-\t" + "0,102,48\t" + "0,153,72\t" +"0,204,96\t" +"217,0,0\t" +"18,41,84\t"+ "27,62,126\t" +"36,82,168\t" +"45,103,210\t" +"87,133,219\t" + "129,164,228\t"
    
    ## 3 colors (red, blue, green):
    color_description = "-\t" + "28,69,139\t" + "28,69,139\t" + "28,69,139\t" + "28,69,139\t" + "28,69,139\t" + "28,69,139\t" + "0,161,75\t" + "0,161,75\t" + "0,161,75\t" + "217,0,0\t"

    header = "label"
    i = 0
    for label in network.structural_labels:
        column_order = column_order + "\t" + str(i)
        header = header + "\t" + label
        i += 1
    data = column_order + "\n" + color_description + "\n" + header + "\n"
        
    for idx, line in enumerate(odTable):
        data = data + network.structural_labels[idx] + "\t" + "\t".join([str(i) for i in line]) + "\n"

    circos_file.write(data)
    circos_file.close()
    
    
def show_subroute_demand(estim_param, evaluate, estpar_ref, eval_ref):
    '''
    Shows the subroutes demand
    '''
    
    plt.figure()
    plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing], rotation=90)
    plt.plot(evaluate.sub_dem_meas_agg_space_mean, 'b', label='measured (VS)')
    if not evaluate.sub_dem_meas_agg_space_std is None:
        plt.plot(evaluate.sub_dem_meas_agg_space_mean + evaluate.sub_dem_meas_agg_space_std, 'b:', linewidth=0.5)
        plt.plot(evaluate.sub_dem_meas_agg_space_mean - evaluate.sub_dem_meas_agg_space_std, 'b:', linewidth=0.5)
        
    if estpar_ref is not None:
        plt.plot(eval_ref.sub_dem_est_agg_space_mean, 'r', label=estpar_ref.case_study_name)
        if not eval_ref.demand_std is None:
            plt.plot(eval_ref.sub_dem_est_agg_space_mean + eval_ref.sub_dem_est_agg_space_std, 'r:', linewidth=0.5)
            plt.plot(eval_ref.sub_dem_est_agg_space_mean - eval_ref.sub_dem_est_agg_space_std, 'r:', linewidth=0.5)
        
    plt.plot(evaluate.sub_dem_est_agg_space_mean, 'g', label=estim_param.case_study_name)
    
    if not evaluate.demand_std is None:
        plt.plot(evaluate.sub_dem_est_agg_space_mean + evaluate.sub_dem_est_agg_space_std, 'g:', linewidth=0.5)
        plt.plot(evaluate.sub_dem_est_agg_space_mean - evaluate.sub_dem_est_agg_space_std, 'g:', linewidth=0.5)
    plt.title('Total demand along VS subroutes')
    plt.legend(loc='upper left')
    
    if estim_param.plot_show:
        plt.show()
    
    _, ymax = plt.ylim()
    plt.ylim((0, ymax))
    
    plt.savefig(estim_param.path_plots + "demand_subroutes" + estim_param.file_format)
    if estim_param.LaTeX:
        tikz_save(estim_param.path_plots + 'demand_subroutes.tex', show_info=False)

        
def show_total_demand(estim_param, evaluate, estpar_ref, eval_ref):
    
    plt.figure()
    plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing], rotation=90)
    if estpar_ref is not None:
        plt.plot(eval_ref.demand_agg_space_mean, 'r', label=estpar_ref.case_study_name)
        if not eval_ref.demand_std is None:
            plt.plot(eval_ref.demand_agg_space_mean + eval_ref.demand_agg_space_std, 'r:', linewidth=1.0)
            plt.plot(eval_ref.demand_agg_space_mean - eval_ref.demand_agg_space_std, 'r:', linewidth=1.0)
           
    plt.plot(evaluate.demand_agg_space_mean, 'g', label=estim_param.case_study_name)
    if not evaluate.demand_std is None:
        plt.plot(evaluate.demand_agg_space_mean + evaluate.demand_agg_space_std, 'g:', linewidth=1.0)
        plt.plot(evaluate.demand_agg_space_mean - evaluate.demand_agg_space_std, 'g:', linewidth=1.0)
    plt.title('Total demand')
    plt.legend(loc='upper left')
    
    if estim_param.plot_show:
        plt.show()
    
    _, ymax = plt.ylim()
    plt.ylim((0, ymax))
    
    plt.savefig(estim_param.path_plots + "total_demand" + estim_param.file_format)
    if estim_param.LaTeX:
        tikz_save(estim_param.path_plots + 'total_demand.tex', show_info=False)
    

def show_occupation(estim_param, evaluate, estpar_ref, eval_ref):
    '''
    Plots the estimated and measured occupation for both PUs of Lausanne Gare:
    '''
    #PUWest:
    plt.subplot(1, 2, 1)
    plt.title("Occupation PU West")
    plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing], rotation=90)
    
    plt.plot(evaluate.occ_meas_mean[::2], 'b', label='measured (VS)')
    if not evaluate.occ_meas_std is None:
        plt.plot(evaluate.occ_meas_mean[::2] + evaluate.occ_meas_std[::2], 'b:', linewidth=0.5)
        plt.plot(evaluate.occ_meas_mean[::2] - evaluate.occ_meas_std[::2], 'b:', linewidth=0.5)
    
    if estpar_ref is not None:
        plt.plot(eval_ref.occ_est_mean[::2], 'r', label=estpar_ref.case_study_name)
        if not eval_ref.demand_std is None:
            plt.plot(eval_ref.occ_est_mean[::2] + eval_ref.occ_est_std[::2], 'r:', linewidth=0.5)
            plt.plot(eval_ref.occ_est_mean[::2] - eval_ref.occ_est_std[::2], 'r:', linewidth=0.5)
    
    plt.plot(evaluate.occ_est_mean[::2], 'g', label=estim_param.case_study_name)
    if not evaluate.demand_std is None:
        plt.plot(evaluate.occ_est_mean[::2] + evaluate.occ_est_std[::2], 'g:', linewidth=0.5)
        plt.plot(evaluate.occ_est_mean[::2] - evaluate.occ_est_std[::2], 'g:', linewidth=0.5)
  
    plt.legend(loc='upper left')
      
    #PUEast:
    plt.subplot(1, 2, 2)
    plt.title("Occupation PU East")
    plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing], rotation=90)
    
    plt.plot(evaluate.occ_meas_mean[1::2], 'b', label='measured (VS)')
    if not evaluate.occ_meas_std is None:
        plt.plot(evaluate.occ_meas_mean[1::2] + evaluate.occ_meas_std[1::2], 'b:', linewidth=0.5)
        plt.plot(evaluate.occ_meas_mean[1::2] - evaluate.occ_meas_std[1::2], 'b:', linewidth=0.5)
    
    if estpar_ref is not None:
        plt.plot(eval_ref.occ_est_mean[1::2], 'r', label=estpar_ref.case_study_name)
        if not eval_ref.demand_std is None:
            plt.plot(eval_ref.occ_est_mean[1::2] + eval_ref.occ_est_std[1::2], 'r:', linewidth=0.5)
            plt.plot(eval_ref.occ_est_mean[1::2] - eval_ref.occ_est_std[1::2], 'r:', linewidth=0.5)
    
    plt.plot(evaluate.occ_est_mean[1::2], 'g', label=estim_param.case_study_name)
    if not evaluate.demand_std is None:
        plt.plot(evaluate.occ_est_mean[1::2] + evaluate.occ_est_std[1::2], 'g:', linewidth=0.5)
        plt.plot(evaluate.occ_est_mean[1::2] - evaluate.occ_est_std[1::2], 'g:', linewidth=0.5)
    
    #plt.legend(loc='upper left')
      
    #plt.suptitle(estim_param.date)
 
    _, ymax = plt.ylim()
    plt.ylim((0, ymax))
 
    plt.savefig(estim_param.path_plots + "occupation_PUs" + estim_param.file_format)
    if estim_param.LaTeX:
        tikz_save(estim_param.path_plots + 'occupation_PUs.tex', show_info=False)
    
    if estim_param.plot_show:
        plt.show()
        
        
def compare_ASE_VS(estim_param, network, evaluate, estpar_ref, eval_ref):
    
    def plot_figure_VS_ASE(estim_param, estpar_ref, network, evaluate, ASE_flow, ASE_flow_std, ASE_link_0, ASE_link_1, VS_flow, VS_flow_std, VS_node, estimated_flow_ref, estimated_flow_ref_std, estimated_flow, estimated_flow_std):
                
        plt.figure()
        
        tot_ASE_flow = np.sum(ASE_flow)
        plt.plot(ASE_flow, 'c', label=network.ASE_edge_names_dict_rev[ASE_link_0] + ' + ' + network.ASE_edge_names_dict_rev[ASE_link_1] + ' (' + str(tot_ASE_flow) + ' ped)')
        if ASE_flow_std is not None:
            plt.plot(ASE_flow + ASE_flow_std, 'c:', linewidth=1.0)
            plt.plot(ASE_flow - ASE_flow_std, 'c:', linewidth=1.0)
    
        if estpar_ref is not None:
            #tot_estimated_flow_ref = np.sum(estimated_flow_ref)
            plt.plot(estimated_flow_ref, 'r', label=estpar_ref.case_study_name + " " + str(ASE_link_0) + ' + ' + str(ASE_link_1))
            if estimated_flow_ref_std is not None:
                plt.plot(estimated_flow_ref + estimated_flow_ref_std, 'r:', linewidth=1.0)
                plt.plot(estimated_flow_ref - estimated_flow_ref_std, 'r:', linewidth=1.0)
    
        #tot_estimated_flow = np.sum(estimated_flow)
        plt.plot(estimated_flow, 'g', label=estim_param.case_study_name + " " + str(ASE_link_0) + ' + ' + str(ASE_link_1))
        if estimated_flow_std is not None:
            plt.plot(estimated_flow + estimated_flow_std, 'g:', linewidth=1.0)
            plt.plot(estimated_flow - estimated_flow_std, 'g:', linewidth=1.0)
    
        tot_VS_flow = np.sum(VS_flow)
        plt.plot(VS_flow, 'b', label="VisioSafe " + str(VS_node) + ' (' + str(tot_VS_flow) + ' ped)')
        if VS_flow_std is not None:
            plt.plot(VS_flow + VS_flow_std, 'b:', linewidth=1.0)
            plt.plot(VS_flow - VS_flow_std, 'b:', linewidth=1.0)

        plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing], rotation=90)
    
        plt.legend(loc='upper left')
        
        _, ymax = plt.ylim()
        plt.ylim((0, ymax))
        
        plt.savefig(estim_param.path_plots + "VSvsASE_" + network.ASE_edge_names_dict_rev[ASE_link_0] + '_' + network.ASE_edge_names_dict_rev[ASE_link_1] + estim_param.file_format)
        if estim_param.LaTeX:
            tikz_save(estim_param.path_plots + "VSvsASE_" + network.ASE_edge_names_dict_rev[ASE_link_0] + '_' + network.ASE_edge_names_dict_rev[ASE_link_1] + ".tex", show_info=False)
    
        if estim_param.plot_show:
            plt.show()
            
        plt.close()
    
        np.savetxt(estim_param.path_textout + "VSvsASE_" + network.ASE_edge_names_dict_rev[ASE_link_0] + '_' + network.ASE_edge_names_dict_rev[ASE_link_1] + ".txt", np.column_stack([VS_flow, ASE_flow]), delimiter=',')
    
    #inflow edges covered by VS and ASE
    ASEVS_origin_flow_dict = {
        #'1wc': ('1c', '1wc'), MALFUNCTIONING, taken out!!
        #'nw': ('nw', '1w'),
        #'nwm': ('nwm', '1w'),
        #'ne': ('ne', '1e'),
        #'nem': ('nem', '1e')
        '1wh': [('nw', 'nwh'), ('nwm', 'nwh')],
        '1e': [('ne', '1eh'), ('nem', '1eh')]
    }
    
    #outflow edges covered by VS and ASE
    ASEVS_destination_flow_dict = {
        #'1wc': ('1c','1h3'), MALFUNCTIONING
        #'nw': ('nw','NW'),
        #'nwm': ('nwm','NWM'),
        #'ne': ('ne','neh2'),
        #'nem': ('nem','NEM')
        '1wh': [('nw', 'NW'), ('nwm', 'NWM')],
        '1e': [('ne', 'neh2'), ('nem', 'NEM')]
    }
    
    error_ASE_VS = np.zeros(len(estim_param.tint_eval_dict) * (len(ASEVS_origin_flow_dict) + len(ASEVS_destination_flow_dict)))
    scatter_VS_ASE = np.zeros((len(estim_param.tint_eval_dict) * (len(ASEVS_origin_flow_dict) + len(ASEVS_destination_flow_dict)), 2))
    err_index = 0

    for VS_node in ASEVS_origin_flow_dict.keys():
        
        ASE_links_in = ASEVS_origin_flow_dict[VS_node]
        
        ASE_link_in_0 = ASE_links_in[0]
        ASE_edge_id_in_0 = network.edges_ASE_dict_rev[ASE_link_in_0]
        edge_id_in_0 = network.edges_dict_rev[ASE_link_in_0]
        
        ASE_link_in_1 = ASE_links_in[1]
        ASE_edge_id_in_1 = network.edges_ASE_dict_rev[ASE_link_in_1]
        edge_id_in_1 = network.edges_dict_rev[ASE_link_in_1]
        
        ASE_link_outs = ASEVS_destination_flow_dict[VS_node]
        ASE_link_out_0 = ASE_link_outs[0]
        ASE_edge_id_out_0 = network.edges_ASE_dict_rev[ASE_link_out_0]
        edge_id_out_0 = network.edges_dict_rev[ASE_link_out_0]
        
        ASE_link_out_1 = ASE_link_outs[1]
        ASE_edge_id_out_1 = network.edges_ASE_dict_rev[ASE_link_out_1]
        edge_id_out_1 = network.edges_dict_rev[ASE_link_out_1]
        
        VS_node_id = network.VS_nodes_dict_rev[VS_node]
    
        ASE_flow_in = np.zeros(len(estim_param.tint_eval_dict))
        ASE_flow_out = np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.ASE_flow_meas_std is not None:
            ASE_flow_in_std = np.zeros(len(estim_param.tint_eval_dict))
            ASE_flow_out_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            ASE_flow_in_std = None
            ASE_flow_out_std = None
        
        if estpar_ref is not None:
            estimated_flow_in_ref = np.zeros(len(estim_param.tint_eval_dict))
            estimated_flow_out_ref = np.zeros(len(estim_param.tint_eval_dict))
            if eval_ref.demand_std is not None:
                estimated_flow_in_ref_std = np.zeros(len(estim_param.tint_eval_dict))
                estimated_flow_out_ref_std = np.zeros(len(estim_param.tint_eval_dict))
            else:
                estimated_flow_in_ref_std = None
                estimated_flow_out_ref_std = None
        else:
            estimated_flow_in_ref = None
            estimated_flow_out_ref = None
            estimated_flow_in_ref_std = None
            estimated_flow_out_ref_std = None
                
        estimated_flow_in = np.zeros(len(estim_param.tint_eval_dict))
        estimated_flow_out = np.zeros(len(estim_param.tint_eval_dict))
        if not evaluate.demand_std is None:
            estimated_flow_in_std = np.zeros(len(estim_param.tint_eval_dict))
            estimated_flow_out_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            estimated_flow_in_std = None
            estimated_flow_out_std = None
            
        VS_flow_in = np.zeros(len(estim_param.tint_eval_dict))
        VS_flow_out = np.zeros(len(estim_param.tint_eval_dict))
        if not evaluate.VS_inflow_meas_std is None:
            VS_flow_in_std = np.zeros(len(estim_param.tint_eval_dict))
            VS_flow_out_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            VS_flow_in_std = None
            VS_flow_out_std = None
            
        
        for t in estim_param.tint_eval_dict.keys():
            ASE_flow_in[t] = evaluate.ASE_flow_meas_mean[t*len(network.edges_ASE)+ASE_edge_id_in_0] + evaluate.ASE_flow_meas_mean[t*len(network.edges_ASE)+ASE_edge_id_in_1]
            ASE_flow_out[t] = evaluate.ASE_flow_meas_mean[t*len(network.edges_ASE)+ASE_edge_id_out_0] + evaluate.ASE_flow_meas_mean[t*len(network.edges_ASE)+ASE_edge_id_out_1]
            if not evaluate.ASE_flow_meas_std is None:
                ASE_flow_in_std[t] = (evaluate.ASE_flow_meas_std[t*len(network.edges_ASE)+ASE_edge_id_in_0] + evaluate.ASE_flow_meas_std[t*len(network.edges_ASE)+ASE_edge_id_in_1])/2
                ASE_flow_out_std[t] = (evaluate.ASE_flow_meas_std[t*len(network.edges_ASE)+ASE_edge_id_out_0] + evaluate.ASE_flow_meas_std[t*len(network.edges_ASE)+ASE_edge_id_out_1])/2
            
            if estpar_ref is not None:
                estimated_flow_in_ref[t] = eval_ref.flow_mean[t*len(network.edges_dict) + edge_id_in_0] + eval_ref.flow_mean[t*len(network.edges_dict) + edge_id_in_1]
                estimated_flow_out_ref[t] = eval_ref.flow_mean[t*len(network.edges_dict) + edge_id_out_0] + eval_ref.flow_mean[t*len(network.edges_dict) + edge_id_out_1]
                if not eval_ref.demand_std is None:
                    estimated_flow_in_ref_std[t] = (eval_ref.flow_std[t*len(network.edges_dict) + edge_id_in_0] + eval_ref.flow_std[t*len(network.edges_dict) + edge_id_in_1])/2
                    estimated_flow_out_ref_std[t] = (eval_ref.flow_std[t*len(network.edges_dict) + edge_id_out_0] + eval_ref.flow_std[t*len(network.edges_dict) + edge_id_out_1])/2
                    
            estimated_flow_in[t] = evaluate.flow_mean[t*len(network.edges_dict) + edge_id_in_0] + evaluate.flow_mean[t*len(network.edges_dict) + edge_id_in_1]
            estimated_flow_out[t] = evaluate.flow_mean[t*len(network.edges_dict) + edge_id_out_0] + evaluate.flow_mean[t*len(network.edges_dict) + edge_id_in_1]
            if not evaluate.demand_std is None:
                estimated_flow_in_std[t] = (evaluate.flow_std[t*len(network.edges_dict) + edge_id_in_0] + evaluate.flow_std[t*len(network.edges_dict) + edge_id_in_1])/2
                estimated_flow_out_std[t] = (evaluate.flow_std[t*len(network.edges_dict) + edge_id_out_0] + evaluate.flow_std[t*len(network.edges_dict) + edge_id_out_1])/2
                
            VS_flow_in[t] = evaluate.VS_inflow_meas_mean[t*len(network.VS_nodes_dict) + VS_node_id]
            VS_flow_out[t] = evaluate.VS_outflow_approx_meas_mean[t*len(network.VS_nodes_dict) + VS_node_id]
            if evaluate.VS_inflow_meas_std is not None:
                VS_flow_in_std[t] = evaluate.VS_inflow_meas_std[t*len(network.VS_nodes_dict) + VS_node_id]
                VS_flow_out_std[t] = evaluate.VS_outflow_approx_meas_std[t*len(network.VS_nodes_dict) + VS_node_id]
  
            error_ASE_VS[err_index] = ASE_flow_in[t] - VS_flow_in[t]
            error_ASE_VS[err_index+1] = ASE_flow_out[t] - VS_flow_out[t]
            scatter_VS_ASE[err_index,:] = [VS_flow_in[t], ASE_flow_in[t]]
            scatter_VS_ASE[err_index+1,:] = [VS_flow_out[t], ASE_flow_out[t]]
            err_index += 2
        
        plot_figure_VS_ASE(estim_param, estpar_ref, network, evaluate, ASE_flow_in, ASE_flow_in_std, ASE_link_in_0, ASE_link_in_1, VS_flow_in, VS_flow_in_std, VS_node, estimated_flow_in_ref, estimated_flow_in_ref_std, estimated_flow_in, estimated_flow_in_std)
        
        plot_figure_VS_ASE(estim_param, estpar_ref, network, evaluate, ASE_flow_out, ASE_flow_out_std, ASE_link_out_0,  ASE_link_out_1, VS_flow_out, VS_flow_out_std, VS_node, estimated_flow_out_ref, estimated_flow_out_ref_std, estimated_flow_out, estimated_flow_out_std)
        
    np.savetxt(estim_param.path_textout + "VSvsASE_scatter" + ".txt", scatter_VS_ASE, delimiter = ',')
       
    # show overall scatter plot
    plt.figure()
    plt.plot(scatter_VS_ASE[:,0],scatter_VS_ASE[:,1],'x')
    plt.xlabel('VisioSafe (ped/min)')
    plt.ylabel('ASE (ped/min)')
    plot_range = [0,scatter_VS_ASE.max()]
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.plot(plot_range,plot_range,'k--')
    
    plt.savefig(estim_param.path_plots + "VSvsASE_scatter" + estim_param.file_format)
    if estim_param.LaTeX:
        tikz_save(estim_param.path_plots + "VSvsASE_scatter" + ".tex", show_info = False)

    if estim_param.plot_show:
        plt.show()
    plt.close()
        
    return error_ASE_VS
        
def compare_TINF_VS(estim_param, network, evaluate, estpar_ref, eval_ref):

    #TINF edges covered by VS
    TINFVS_edges_dict = {
        '34d': ('34d', '34w'),
        '56d': ('56d', '56w'), 
        '78d': ('78d', '78w'), 
        '34c': ('34c', '34w'),
        '56c': ('56c', '56w'), 
        '78c': ('78c', '78w'), 
        '34b': ('34b', '34e'), 
        '56b': ('56b', '56e'), 
        '34a': ('34a', '34e'), 
        '56a': ('56a', '56e')}
    
    error_TINF_VS = np.zeros(len(estim_param.tint_eval_dict)*len(TINFVS_edges_dict))
    err_index = 0;

    for VS_node, TINF_link in TINFVS_edges_dict.iteritems():
        TINF_edge_id = network.edges_TINF_dict_rev[TINF_link]
        
        edge_id = network.edges_dict_rev[TINF_link]
        
        VS_node_id = network.VS_nodes_dict_rev[VS_node]
        
        TINF_flow=np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.TINF_flow_prior_std is not None:
            TINF_flow_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            TINF_flow_std = None
        
        if estpar_ref is not None:
            estimated_flow_ref=np.zeros(len(estim_param.tint_eval_dict))
            if eval_ref.flow_std is not None:
                estimated_flow_ref_std = np.zeros(len(estim_param.tint_eval_dict))
            else:
                estimated_flow_ref_std = None
        
        estimated_flow=np.zeros(len(estim_param.tint_eval_dict))
        if not evaluate.flow_std is None:
            estimated_flow_std=np.zeros(len(estim_param.tint_eval_dict))
        else:
            estimated_flow_std = None
            
        VS_flow=np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.VS_inflow_meas_std is not None:
            VS_flow_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            VS_flow_std = None
        
        for t in estim_param.tint_eval_dict.keys():
            TINF_flow[t] = evaluate.TINF_flow_prior_mean[t*len(network.edges_TINF)+TINF_edge_id]
            if TINF_flow_std is not None:
                TINF_flow_std[t] = evaluate.TINF_flow_prior_std[t*len(network.edges_TINF)+TINF_edge_id]
            
            if estpar_ref is not None:
                estimated_flow_ref[t] = eval_ref.flow_mean[t*len(network.edges_dict) + edge_id]
                if estimated_flow_ref_std is not None:
                    estimated_flow_ref_std[t] = eval_ref.flow_std[t*len(network.edges_dict) + edge_id]
                                
            estimated_flow[t] = evaluate.flow_mean[t*len(network.edges_dict) + edge_id]
            if evaluate.demand_std is not None:
                estimated_flow_std[t] = evaluate.flow_std[t*len(network.edges_dict) + edge_id]
                
            VS_flow[t] = evaluate.VS_inflow_meas_mean[t*len(network.VS_nodes_dict) + VS_node_id]
            if VS_flow_std is not None:
                VS_flow_std[t] = evaluate.VS_inflow_meas_std[t*len(network.VS_nodes_dict) + VS_node_id]
            
            error_TINF_VS[err_index] = TINF_flow[t] - VS_flow[t]
            err_index += 1
        
        plt.figure()
        plt.plot(VS_flow, 'b', label="VisioSafe " + str(VS_node))
        if VS_flow_std is not None:
            plt.plot(VS_flow+VS_flow_std, 'b:', linewidth = 1.0)
            plt.plot(VS_flow-VS_flow_std, 'b:', linewidth = 1.0)
            
        plt.plot(TINF_flow, 'c', label="TINF "+ network.edges_TINF_origins_dict[TINF_edge_id])
        if TINF_flow_std is not None:
            plt.plot(TINF_flow + TINF_flow_std, 'c:', linewidth = 1.0)
            plt.plot(TINF_flow - TINF_flow_std, 'c:', linewidth = 1.0)
            
        if estpar_ref is not None:
            plt.plot(estimated_flow_ref, 'r', label=estpar_ref.case_study_name + " " + str(TINF_link))
            if estimated_flow_ref_std is not None:
                plt.plot(estimated_flow_ref + estimated_flow_ref_std, 'r:', linewidth = 1.0)
                plt.plot(estimated_flow_ref - estimated_flow_ref_std, 'r:', linewidth = 1.0)
            
        plt.plot(estimated_flow, 'g', label=estim_param.case_study_name + " " + str(TINF_link))
        if estimated_flow_std is not None:
            plt.plot(estimated_flow + estimated_flow_std, 'g:', linewidth = 1.0)
            plt.plot(estimated_flow - estimated_flow_std, 'g:', linewidth = 1.0)
            
        plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing],rotation=90)
        plt.legend(loc='upper left')
        
        _, ymax = plt.ylim()
        plt.ylim((0,ymax))
        
        plt.savefig(estim_param.path_plots + "VSvsTINF_" + network.edges_TINF_origins_dict[TINF_edge_id] + estim_param.file_format)
        if estim_param.LaTeX:
            tikz_save(estim_param.path_plots + "VSvsTINF_" + network.edges_TINF_origins_dict[TINF_edge_id] + ".tex", show_info = False)
    
        if estim_param.plot_show:
            plt.show()
            
        plt.close()
        
    return error_TINF_VS


def show_TINF(estim_param, network, evaluate, estpar_ref, eval_ref):

    for TINF_edge_id in network.edges_TINF_dict.keys():

        TINF_link = network.edges_TINF_dict[TINF_edge_id]
        edge_id = network.edges_dict_rev[TINF_link]
    
        TINF_flow=np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.TINF_flow_prior_std is not None:
            TINF_flow_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            TINF_flow_std = None

        if estpar_ref is not None:
            estimated_flow_ref=np.zeros(len(estim_param.tint_eval_dict))
            if eval_ref.flow_std is not None:
                estimated_flow_ref_std = np.zeros(len(estim_param.tint_eval_dict))
            else:
                estimated_flow_ref_std = None
        
        estimated_flow=np.zeros(len(estim_param.tint_eval_dict))
        if not evaluate.flow_std is None:
            estimated_flow_std=np.zeros(len(estim_param.tint_eval_dict))
        else:
            estimated_flow_std = None

        
        for t in estim_param.tint_eval_dict.keys():
            TINF_flow[t] = evaluate.TINF_flow_prior_mean[t*len(network.edges_TINF)+TINF_edge_id]
            if TINF_flow_std is not None:
                TINF_flow_std[t] = evaluate.TINF_flow_prior_std[t*len(network.edges_TINF)+TINF_edge_id]
            
            if estpar_ref is not None:
                estimated_flow_ref[t] = eval_ref.flow_mean[t*len(network.edges_dict) + edge_id]
                if estimated_flow_ref_std is not None:
                    estimated_flow_ref_std[t] = eval_ref.flow_std[t*len(network.edges_dict) + edge_id]
                                
            estimated_flow[t] = evaluate.flow_mean[t*len(network.edges_dict) + edge_id]
            if evaluate.demand_std is not None:
                estimated_flow_std[t] = evaluate.flow_std[t*len(network.edges_dict) + edge_id]
            
                    
        plt.figure()
        
        plt.plot(TINF_flow, 'c', label="TINF "+ network.edges_TINF_origins_dict[TINF_edge_id])
        if TINF_flow_std is not None:
            plt.plot(TINF_flow + TINF_flow_std, 'c:', linewidth = 1.0)
            plt.plot(TINF_flow - TINF_flow_std, 'c:', linewidth = 1.0)
            
        if estpar_ref is not None:
            plt.plot(estimated_flow_ref, 'r', label=estpar_ref.case_study_name + " " + str(TINF_link))
            if estimated_flow_ref_std is not None:
                plt.plot(estimated_flow_ref + estimated_flow_ref_std, 'r:', linewidth = 1.0)
                plt.plot(estimated_flow_ref - estimated_flow_ref_std, 'r:', linewidth = 1.0)
            
        plt.plot(estimated_flow, 'g', label=estim_param.case_study_name + " " + str(TINF_link))
        if estimated_flow_std is not None:
            plt.plot(estimated_flow + estimated_flow_std, 'g:', linewidth = 1.0)
            plt.plot(estimated_flow - estimated_flow_std, 'g:', linewidth = 1.0)
        
        
        plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing],rotation=90)
        plt.legend(loc='upper left')
        
        _, ymax = plt.ylim()
        plt.ylim((0,ymax))
        
        plt.savefig(estim_param.path_plots + "TINF_" + network.edges_TINF_origins_dict[TINF_edge_id] + estim_param.file_format)
        if estim_param.LaTeX:
            tikz_save(estim_param.path_plots + "TINF_" + network.edges_TINF_origins_dict[TINF_edge_id] + ".tex", show_info = False)
    
        if estim_param.plot_show:
            plt.show()
            
        plt.close()
        
def show_ASE(estim_param, network, evaluate, estpar_ref, eval_ref):

    for ASE_edge_id in network.edges_ASE_dict.keys():

        ASE_link = network.edges_ASE_dict[ASE_edge_id]
        edge_id = network.edges_dict_rev[ASE_link]
        
        ASE_flow = np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.ASE_flow_meas_std is not None:
            ASE_flow_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            ASE_flow_std = None
        
        if estpar_ref is not None:
            estimated_flow_ref=np.zeros(len(estim_param.tint_eval_dict))
            if eval_ref.demand_std is not None:
                estimated_flow_ref_std = np.zeros(len(estim_param.tint_eval_dict))
            else:
                estimated_flow_ref_std = None
                
        estimated_flow = np.zeros(len(estim_param.tint_eval_dict))
        if not evaluate.demand_std is None:
            estimated_flow_std=np.zeros(len(estim_param.tint_eval_dict))
        else:
            estimated_flow_std = None    
                    
        for t in estim_param.tint_eval_dict.keys():
            
            ASE_flow[t] = evaluate.ASE_flow_meas_mean[t*len(network.edges_ASE)+ASE_edge_id]
            if not evaluate.ASE_flow_meas_std is None:
                ASE_flow_std[t] = evaluate.ASE_flow_meas_std[t*len(network.edges_ASE)+ASE_edge_id]
            
            if estpar_ref is not None:
                estimated_flow_ref[t] = eval_ref.flow_mean[t*len(network.edges_dict) + edge_id]
                if not eval_ref.demand_std is None:
                    estimated_flow_ref_std[t] = eval_ref.flow_std[t*len(network.edges_dict) + edge_id]
                    
            estimated_flow[t] = evaluate.flow_mean[t*len(network.edges_dict) + edge_id]
            if not evaluate.demand_std is None:
                estimated_flow_std[t] = evaluate.flow_std[t*len(network.edges_dict) + edge_id]

        
        plt.figure()
        
        ASE_flow_tot = np.sum(ASE_flow)
        plt.plot(ASE_flow, 'c', label=network.ASE_edge_names_dict_rev[ASE_link] + ' (' + str(ASE_flow_tot) + ' ped)')
        if ASE_flow_std is not None:
            plt.plot(ASE_flow+ASE_flow_std, 'c:', linewidth = 1.0)
            plt.plot(ASE_flow-ASE_flow_std, 'c:', linewidth = 1.0)
    
        if estpar_ref is not None:
            plt.plot(estimated_flow_ref, 'r', label=estpar_ref.case_study_name + " " + str(ASE_link))
            if estimated_flow_ref_std is not None:
                plt.plot(estimated_flow_ref + estimated_flow_ref_std, 'r:', linewidth = 1.0)
                plt.plot(estimated_flow_ref - estimated_flow_ref_std, 'r:', linewidth = 1.0)
    
        plt.plot(estimated_flow, 'g', label=estim_param.case_study_name + " " + str(ASE_link))
        if estimated_flow_std is not None:
            plt.plot(estimated_flow + estimated_flow_std, 'g:', linewidth = 1.0)
            plt.plot(estimated_flow - estimated_flow_std, 'g:', linewidth = 1.0)
    
        plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing],rotation=90)
        plt.legend(loc='upper left')
        
        _, ymax = plt.ylim()
        plt.ylim((0,ymax))
        
        plt.savefig(estim_param.path_plots + network.ASE_edge_names_dict_rev[ASE_link] + estim_param.file_format)
        if estim_param.LaTeX:
            tikz_save(estim_param.path_plots + network.ASE_edge_names_dict_rev[ASE_link] + ".tex", show_info = False)
    
        if estim_param.plot_show:
            plt.show()
            
        plt.close()
        
def show_VS_flows(estim_param, network, evaluate, estpar_ref, eval_ref):
    
    # plot; flag = 'in', 'out' (for text labels only)
    def plot_VS_flow(estim_param, estpar_ref, evaluate, VS_flow, VS_flow_std, est_flow, est_flow_std, est_flow_ref, est_flow_ref_std, flow_edge, VS_node, flag):
    
        plt.figure()
        plt.plot(VS_flow, 'b', label=flag + "flow VS " + str(VS_node))
        if VS_flow_std is not None:
            plt.plot(VS_flow + VS_flow_std, 'b:', linewidth = 1.0)
            plt.plot(VS_flow - VS_flow_std, 'b:', linewidth = 1.0)
        
        if estpar_ref is not None:
            plt.plot(est_flow_ref, 'r', label=estpar_ref.case_study_name + " " + str(flow_edge))
            if est_flow_ref_std is not None:
                plt.plot(est_flow_ref + est_flow_ref_std, 'r:', linewidth = 1.0)
                plt.plot(est_flow_ref - est_flow_ref_std, 'r:', linewidth = 1.0)
            
        plt.plot(est_flow, 'g', label=estim_param.case_study_name + " " + str(flow_edge))
        if est_flow_std is not None:
            plt.plot(est_flow + est_flow_std, 'g:', linewidth = 1.0)
            plt.plot(est_flow - est_flow_std, 'g:', linewidth = 1.0)
            
        plt.xticks(estim_param.tint_eval_dict.keys()[::estim_param.axis_ticks_spacing], evaluate.axis_ticks_timestamps[::estim_param.axis_ticks_spacing],rotation=90)
        
        plt.legend(loc='upper left')
        
        _, ymax = plt.ylim()
        plt.ylim((0,ymax))
        
        plt.savefig(estim_param.path_plots + "VS_" + flag + "flow_" + VS_node + estim_param.file_format)
        if estim_param.LaTeX:
            tikz_save(estim_param.path_plots + "VS_" + flag + "flow_" + VS_node + ".tex", show_info = False)
    
        if estim_param.plot_show:
            plt.show()
            
        plt.close()

    for node_key in network.VS_nodes_dict.keys():

        VS_node = network.VS_nodes_dict[node_key]

        inflow_edge = network.VS_inflow_edges_dict[node_key]
        outflow_edge = network.VS_outflow_edges_dict[node_key]
        
        VS_in = np.zeros(len(estim_param.tint_eval_dict))
        VS_out_approx = np.zeros(len(estim_param.tint_eval_dict))
        if evaluate.VS_inflow_meas_std is not None:
            VS_in_std = np.zeros(len(estim_param.tint_eval_dict))
            VS_out_approx_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            VS_in_std = None
            VS_out_approx_std = None

        if estpar_ref is not None:
            est_in_ref = np.zeros(len(estim_param.tint_eval_dict))
            est_out_ref = np.zeros(len(estim_param.tint_eval_dict))
            if eval_ref.demand_std is not None:
                est_in_ref_std = np.zeros(len(estim_param.tint_eval_dict))
                est_out_ref_std = np.zeros(len(estim_param.tint_eval_dict))
            else:
                est_in_ref_std = None
                est_out_ref_std = None
        else:
            est_in_ref = None
            est_out_ref = None
            est_in_ref_std = None
            est_out_ref_std = None

        est_in = np.zeros(len(estim_param.tint_eval_dict))
        est_out = np.zeros(len(estim_param.tint_eval_dict))
        
        if evaluate.demand_std is not None:
            est_in_std = np.zeros(len(estim_param.tint_eval_dict))
            est_out_std = np.zeros(len(estim_param.tint_eval_dict))
        else:
            est_in_std = None
            est_out_std = None
        
        for t in estim_param.tint_eval_dict.keys():

            VS_in[t] = evaluate.VS_inflow_meas_mean[t * len(network.VS_nodes_dict) + node_key]
            VS_out_approx[t] = evaluate.VS_outflow_approx_meas_mean[t * len(network.VS_nodes_dict) + node_key]
            if VS_in_std is not None:
                VS_in_std[t] = evaluate.VS_inflow_meas_std[t * len(network.VS_nodes_dict) + node_key]
                VS_out_approx_std[t] = evaluate.VS_outflow_approx_meas_std[t * len(network.VS_nodes_dict) + node_key]
                
            if estpar_ref is not None:
                est_in_ref[t] = eval_ref.VS_inflow_est_mean[t * len(network.VS_nodes_dict) + node_key]
                est_out_ref[t] = eval_ref.VS_outflow_est_mean[t * len(network.VS_nodes_dict) + node_key]
                if est_in_ref_std is not None:
                    est_in_ref_std[t] = eval_ref.VS_inflow_est_std[t * len(network.VS_nodes_dict) + node_key]
                    est_out_ref_std[t] = eval_ref.VS_outflow_est_std[t * len(network.VS_nodes_dict) + node_key]

            est_in[t] = evaluate.VS_inflow_est_mean[t * len(network.VS_nodes_dict) + node_key]
            est_out[t] = evaluate.VS_outflow_est_mean[t * len(network.VS_nodes_dict) + node_key]
            if est_in_std is not None:
                est_in_std[t] = evaluate.VS_inflow_est_std[t * len(network.VS_nodes_dict) + node_key]
                est_out_std[t] = evaluate.VS_outflow_est_std[t * len(network.VS_nodes_dict) + node_key]
        
        plot_VS_flow(estim_param, estpar_ref, evaluate, VS_in, VS_in_std, est_in, est_in_std, est_in_ref, est_in_ref_std, inflow_edge, VS_node, 'in')
        plot_VS_flow(estim_param, estpar_ref, evaluate, VS_out_approx, VS_out_approx_std, est_out, est_out_std, est_out_ref, est_out_ref_std, outflow_edge, VS_node, 'out')

        
def show_scatter_plots(estim_param, evaluate, estpar_ref, eval_ref):

    def scatter_plot(estim_param, vec, vec_ref, yerr, xerr, tit, tit_ref, legend, file_name):
        plt.figure()
        if xerr is None:
            plt.errorbar(vec_ref, vec, yerr, linestyle='None', marker='^', label=legend + " (ped/min)")
        else:
            plt.errorbar(vec_ref, vec, yerr, xerr, linestyle='None', marker='^', label=legend + " (ped/min)")
        plt.xlabel(tit_ref)
        plt.ylabel(tit)
        plot_range = [0, max(vec_ref.max(), vec.max())]
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.plot(plot_range, plot_range, 'k--')

        plt.savefig(estim_param.path_plots + "scatter_" + file_name + estim_param.file_format)
        '''
        if estim_param.LaTeX:
                tikz_save(estim_param.path_plots + "scatter_" + file_name + ".tex", show_info = False)
        '''
        if estim_param.plot_show:
                plt.show()
        plt.close()

    if estpar_ref is not None:
        # estim_param, vec, vec_ref, tit, tit_ref, legend, file_name
        scatter_plot(estim_param, eval_ref.demand_mean, evaluate.demand_mean, evaluate.demand_std, eval_ref.demand_std, estpar_ref.case_study_name, estim_param.case_study_name, 'disaggregate demand_mean', 'demand_disagg')
    
        scatter_plot(estim_param, eval_ref.demand_agg_time_mean, evaluate.demand_agg_time_mean, evaluate.demand_agg_time_std, eval_ref.demand_agg_time_std, estpar_ref.case_study_name, estim_param.case_study_name, 'temp. agg. demand_mean', 'demand_agg_time_mean')
    
        scatter_plot(estim_param, eval_ref.demand_agg_space_mean, evaluate.demand_agg_space_mean, evaluate.demand_agg_space_std, eval_ref.demand_agg_space_std, estpar_ref.case_study_name, estim_param.case_study_name, 'spatially agg. demand_mean', 'demand_agg_space_mean')

    scatter_plot(estim_param, evaluate.occ_meas_mean, evaluate.occ_est_mean, evaluate.occ_est_std, evaluate.occ_meas_std, 'measured', 'estimated', 'occupation disagg', 'occ_disagg')

    scatter_plot(estim_param, evaluate.sub_dem_meas_mean, evaluate.sub_dem_est_mean, evaluate.sub_dem_est_std, evaluate.sub_dem_meas_std, 'measured', 'estimated', 'disagg subr demand_mean', 'subr_demand_disagg')

    scatter_plot(estim_param, evaluate.sub_dem_meas_agg_time_mean, evaluate.sub_dem_est_agg_time_mean, evaluate.sub_dem_est_agg_time_std, evaluate.sub_dem_meas_agg_time_std, 'measured', 'estimated', 'temp. aggr. subr demand_mean', 'subr_demand_agg_time')

    scatter_plot(estim_param, evaluate.sub_dem_meas_agg_space_mean, evaluate.sub_dem_est_agg_space_mean, evaluate.sub_dem_est_agg_space_std, evaluate.sub_dem_meas_agg_space_std, 'measured', 'estimated', 'spatially aggr. subr demand_mean', 'subr_demand_agg_space')

    scatter_plot(estim_param, evaluate.ASE_flow_meas_mean, evaluate.ASE_flow_est_mean, evaluate.ASE_flow_est_std, evaluate.ASE_flow_meas_std, 'measured', 'estimated', 'ASE flows', 'ASE_flows')

    scatter_plot(estim_param, evaluate.TINF_flow_prior_mean, evaluate.TINF_flow_est_mean, evaluate.TINF_flow_est_std, evaluate.TINF_flow_prior_std, 'prior', 'estimated', 'TINF flows', 'TINF_flows')
    
    scatter_plot(estim_param, evaluate.VS_inflow_meas_mean, evaluate.VS_inflow_est_mean, evaluate.VS_inflow_est_std, evaluate.VS_inflow_meas_std, 'measured', 'estimated', 'VS inflow', 'VS_in')
    
    scatter_plot(estim_param, evaluate.VS_outflow_approx_meas_mean, evaluate.VS_outflow_est_mean, evaluate.VS_outflow_est_std, evaluate.VS_outflow_approx_meas_std, 'measured', 'estimated', 'VS outflow', 'VS_out')


def write_protocol(estim_param, network, evaluate, estpar_ref, eval_ref, postproc):
   
    file_name = estim_param.path_textout + "log.txt"
    log_file = open(file_name, 'w')
    
    output_stream = "Approx. size of " + estim_param.case_study_name + ": " + str(network.no_variables) + " variables, " + str(network.no_constraints) + " constraints "
    output_stream += estim_param.print_incr_runtime(time_elaps='full', returnString=True) + "\n\n"
    
    def protocol_per_case(estpar, evalu):
        data = "*** " + estpar.case_study_name + " ***\n"
        data += "muZero = " + str(estpar.muZero) + ", muLink = " + str(estpar.muLink) + ", muSubRoute = " + str(estpar.muSubRoute) + ", muTINF = " + str(estpar.muTINF) + ", muHist = " + str(estpar.muHist) + ", muHistAgg = " + str(estpar.muHistAgg) + "\n\n"
        data += "occ_agg_space_err: " + str(evalu.occ_agg_space_err) + "\n\n"
        data += "subroute_err: " + str(evalu.subroute_err) + "\n\n"
        data += "subroute_agg_space_err: " + str(evalu.subroute_agg_space_err) + "\n\n"
        data += "subroute_agg_time_err: " + str(evalu.subroute_agg_time_err) + "\n\n"
        data += "ASE_flow_err: " + str(evalu.ASE_flow_error) + "\n\n"
        data += "TINF_flow_err: " + str(evalu.TINF_flow_error) + "\n\n"
        data += "VS_err_inflow: " + str(evalu.VS_err_inflow) + "\n\n"
        data += "VS_approx_err_outflow: " + str(evalu.VS_approx_err_outflow) + "\n\n"
            
        return data
    
    output_stream += protocol_per_case(estim_param, evaluate)
    
    if estpar_ref is not None:
        output_stream += "*** Relative change in RMSE (" + estim_param.case_study_name + "/" + estpar_ref.case_study_name + " - 1) *** \n"
        output_stream += "occ_agg_space_err: " + str("%.2f" % ((evaluate.occ_agg_space_err['RMSE'] / eval_ref.occ_agg_space_err['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "subroute_err: " + str("%.2f" % ((evaluate.subroute_err['RMSE'] / eval_ref.subroute_err['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "subroute_agg_space_err: " + str("%.2f" % ((evaluate.subroute_agg_space_err['RMSE'] / eval_ref.subroute_agg_space_err['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "subroute_agg_time_err: " + str("%.2f" % ((evaluate.subroute_agg_time_err['RMSE'] / eval_ref.subroute_agg_time_err['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "ASE_flow_err: " + str("%.2f" % ((evaluate.ASE_flow_error['RMSE'] / eval_ref.ASE_flow_error['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "TINF_flow_err: " + str("%.2f" % ((evaluate.TINF_flow_error['RMSE'] / eval_ref.TINF_flow_error['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "VS_err_inflow: " + str("%.2f" % ((evaluate.VS_err_inflow['RMSE'] / eval_ref.VS_err_inflow['RMSE'] - 1) * 100)) + "% \n"
        output_stream += "VS_approx_err_outflow: " + str("%.2f" % ((evaluate.VS_approx_err_outflow['RMSE'] / eval_ref.VS_approx_err_outflow['RMSE'] - 1) * 100)) + "% \n\n"
        output_stream += protocol_per_case(estpar_ref, eval_ref)
    
    if (estim_param.compare_ASE_VS and estim_param.compare_TINF_VS):
        output_stream += "*** Error between TINF, ASE, VS *** \n"
        output_stream += "ASE-VS: mean = " + str(np.mean(postproc.error_ASE_VS)) + ", std = " + str(np.std(postproc.error_ASE_VS)) + "\n"
        output_stream += "TINF-VS: mean = " + str(np.mean(postproc.error_TINF_VS)) + ", std = " + str(np.std(postproc.error_TINF_VS)) + "\n"

    log_file.write(output_stream)
    log_file.close()

    
def generate_output_dir(estim_param):
    # generate output directory
    if not os.path.exists(estim_param.path_output):
        os.makedirs(estim_param.path_output)
    
    # generate output subfolders
    for directory in {estim_param.path_output_plots, estim_param.path_output_text}:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif estim_param.clearOutput:
            shutil.rmtree(directory)
            os.makedirs(directory)
