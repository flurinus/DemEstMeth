'''
@author: quentin, flurin, eduard
'''

import networkx as nx  # http://networkx.github.io
import numpy as np
from collections import defaultdict


class LausNetwork:
    '''
    Class containing the network specifications for the Lausanne station
    '''
    
    def __init__(self, estim_param):
        '''
        Constructor for this network class
        '''
        print "Loading Lausanne network",

        # Calls the method specifing the network
        self.GdL_create_network(estim_param)

        # Calculates the number of constraints
        self.problem_size(estim_param)

        # Prints the time taken for the loading of the network
        estim_param.print_incr_runtime()
        
    def GdL_create_network(self, estim_param):
        '''
        Creates the graph of Gare de Lausanne
        '''
        self.G = nx.MultiDiGraph()

        ''' ---------------------------------------------------
                            Defines the vertices
        ----------------------------------------------------'''
        
        # Dictionary used to plot the graph. Includes all the nodes of the network.
        self.positions = {'NW': (-0.7, 20), 'NWM': (1, 20), 'NE': (7, 20), 'NEM': (9, 20), 'SW': (0, -3), 'SE': (8, -3), 'N': (3, 20), '1D': (-2.5, 14), '1C': (3, 14), 'BAR': (-0.8, 16.1), 'nww': (-1.4, 18), 'KIOSK': (0.9, 12), 'kioskh': (0, 12), 'SHOP': (-1.5, -3), '1h1': (-2, 14), '1h2': (-1.7, 14), '1d': (-0.9, 14), 'nwh': (0, 17), '1wh': (0, 16), '1w': (0, 14), '1wc': (0.9, 14), '1c': (1.4, 14), '1h3': (1.8, 14), '1h4': (2.4, 14), 'nh': (3, 19), 'h': (3, 18), '34B': (5.5, 9), '34C': (3, 9), '34A': (10.5, 9), '34D': (-2.5, 9), '56B': (5.5, 6), '56C': (3, 6), '56D': (-2.5, 6), '56A': (10.5, 6), '78B': (5.5, 3), '78C': (3, 3), '78A': (10.5, 3), '78D': (-2.5, 3), '9C': (3, 0), '9D': (-2.5, 0), 'nw': (-0.7, 18), 'sw': (0, -1.5), 'nwm': (1, 18), 'ne': (7, 16), 'neh1': (7, 19), 'neh2': (7, 18), 'nem': (9, 16), 'se': (8, -1.5), '1eh': (8, 15), '1e': (8, 14), '1AB': (10.5, 14), '70FE': (10.5, 16), '70fe': (9.5, 16), '1ab': (9.5, 14), '34d': (-0.9, 9), '34c': (0.9, 9), '34b': (7.1, 9), '34a': (8.9, 9), '56d': (-0.9, 6), '56c': (0.9, 6), '56b': (7.1, 6), '56a': (8.9, 6), '78d': (-0.9, 3), '78c': (0.9, 3), '78b': (7.1, 3), '78a': (8.9, 3), '9d': (-0.9, 0), '9c': (0.9, 0), '9h1': (-0.6, 0), '9h2': (-0.3, 0), '9h3': (0.3, 0), '9h4': (0, 1), '34w': (0, 9), '56w': (0, 6), '78w': (0, 3), '9w': (0, 0), '34e': (8, 9), '56e': (8, 6), '56h': (8, 4), '78e': (8, 3)}
        
        self.node_labels = {'NW': 'NW', 'NWM': 'NWM', 'NE': 'NE', 'NEM': 'NEM', 'SW': 'SW', 'SE': 'SE', 'N': 'N', '1D': '1D', '1C': '1C', 'BAR': 'BAR', 'nww': 'nww', 'KIOSK': 'KIOSK', 'kioskh': 'kiokh', 'SHOP': 'SHOP', '1h1': '1h1', '1h2': '1h2', '1d': '1d', 'nwh': 'nwh', '1wh': '1wh', '1w': '1w', '1wc': '1wc', '1c': '1c', '1h3': '1h3', '1h4': '1h4', 'nh': 'nh', 'h': 'h', '34B': '34B', '34C': '34C', '34A': '34A', '34D': '34D', '56B': '56B', '56C': '56C', '56D': '56D', '56A': '56A', '78B': '78B', '78C': '78C', '78A': '78A', '78D': '78D', '9C': '9C', '9D': '9D', 'nw': 'nw', 'sw': 'sw', 'nwm': 'nwm', 'ne': 'ne', 'neh1': 'neh1', 'neh2': 'neh2', 'nem': 'nem', 'se': 'se', '1eh': '1eh', '1e': '1e', '1AB': '1AB', '70FE': '70FE', '70fe': '70fe', '1ab': '1ab', '34d': '34d', '34c': '34c', '34b': '34b', '34a': '34a', '56d': '56d', '56c': '56c', '56b': '56b', '56a': '56a', '78d': '78d', '78c': '78c', '78b': '78b', '78a': '78a', '9d': '9d', '9c': '9c', '9h1': '9h1', '9h2': '9h2', '9h3': '9h3', '9h4': '9h4', '34w': '34w', '56w': '56w', '78w': '78w', '9w': '9w', '34e': '34e', '56e': '56e', '56h': '56h', '78e': '78e'}
        self.centroids_labels = {'NW': 'NW', 'NWM': 'NWM', 'NE': 'NE', 'NEM': 'NEM', 'SW': 'SW', 'SE': 'SE', 'N': 'N', '1D': '1D', '1C': '1C', 'BAR': 'BAR', 'KIOSK': 'KIOSK', 'SHOP': 'SHOP', '34B': '34B', '34C': '34C', '34A': '34A', '34D': '34D', '56B': '56B', '56C': '56C', '56D': '56D', '56A': '56A', '78B': '78B', '78C': '78C', '78A': '78A', '78D': '78D', '9C': '9C', '9D': '9D', '1AB': '1AB', '70FE': '70FE'}
        self.not_centroids_labels = {'nww': 'nww', 'kioskh': 'kiokh', '1h1': '1h1', '1h2': '1h2', '1d': '1d', 'nwh': 'nwh', '1wh': '1wh', '1w': '1w', '1wc': '1wc', '1c': '1c', '1h3': '1h3', '1h4': '1h4', 'nh': 'nh', 'h': 'h', 'nw': 'nw', 'sw': 'sw', 'nwm': 'nwm', 'ne': 'ne', 'neh1': 'neh1', 'neh2': 'neh2', 'nem': 'nem', 'se': 'se', '1eh': '1eh', '1e': '1e', '70fe': '70fe', '1ab': '1ab', '34d': '34d', '34c': '34c', '34b': '34b', '34a': '34a', '56d': '56d', '56c': '56c', '56b': '56b', '56a': '56a', '78d': '78d', '78c': '78c', '78b': '78b', '78a': '78a', '9d': '9d', '9c': '9c', '9h1': '9h1', '9h2': '9h2', '9h3': '9h3', '9h4': '9h4', '34w': '34w', '56w': '56w', '78w': '78w', '9w': '9w', '34e': '34e', '56e': '56e', '56h': '56h', '78e': '78e'}

        #We define the nodes, classifying th1em as centroids (platform and not platform), ASE_measurement nodes, VS_measurement_nodes (West and East PU) and other nodes.
        self.centroids = ['NW', 'NWM', 'NE', 'NEM', 'SW', 'SE', 'N', 'KIOSK', 'BAR', 'SHOP', '1D', '1C', '70FE', '1AB', '34D', '34C', '34B', '34A', '56D', '56C', '56B', '56A', '78D', '78C', '78B', '78A', '9D', '9C']
        Centroids_Platform = ['1D', '1C', '70FE', '1AB', '34D', '34C', '34B', '34A', '56D', '56C', '56B', '56A', '78D', '78C', '78B', '78A', '9D', '9C']
        Centroids_No_Platform = ['NW', 'NWM', 'NE', 'NEM', 'SW', 'SE', 'N', 'KIOSK', 'BAR', 'SHOP']
        Centroids_Entrance_Exit = ['NW', 'NWM', 'NE', 'NEM', 'SW', 'SE', 'N']
        Centroids_Shop = ['KIOSK', 'BAR', 'SHOP']
        
        Platforms = {'1', '3/4', '5/6', '7/8', '9', '70'}
        
        Centroid_Types_Detail = ['platform', 'entrance', 'shop']
        Centroid_Types_RCh = ['platform', 'non-platform']
        
        self.ASE_measurement_nodes = ['nw', 'nww', 'nwm', '1c', 'nh', 'neh1', 'ne', 'nem', 'sw', '9h2', '9h3', 'se']
        self.VS_measurement_nodes = ['1h1', '1h4', '70fe', '1ab', '34d', '34c', '34b', '34a', '56d', '56c', '56b', '56a', '78d', '78c', '78b', '78a', '9d', '9c']  # TINF nodes
        self.other_nodes = ['h', '1wc', 'kioskh', '1h2', '1d', 'nwh', '1wh', '1w', '1h3', 'neh2', '1eh', '1e', '9h1', '9h4', '34w', '56w', '78w', '9w', '34e', '56e', '56h', '78e']
        VS_west_nodes = ['1wh', '1d', '1wc', '34d', '34c', '56d', '56c', '78d', '78c', '9h4']
        VS_east_nodes = ['1e', '34b', '34a', '56b', '56a', '56h']
        VS_nodes = VS_west_nodes + VS_east_nodes
        
        #We create dictionaries for the centroids
        self.centroids_dict = {}
        self.centroids_dict_rev = {}
        for idx, node in enumerate(self.centroids):
            self.centroids_dict[idx] = node
            self.centroids_dict_rev[node] = idx

        self.centroids_p_dict = {}
        for idx, node in enumerate(Centroids_Platform):
            self.centroids_p_dict[idx] = node
        
        self.centroids_np_dict = {}
        for idx, node in enumerate(Centroids_No_Platform):
            self.centroids_np_dict[idx] = node
            
        self.centroids_ee_dict = {}
        for idx, node in enumerate(Centroids_Entrance_Exit):
            self.centroids_ee_dict[idx] = node
        
        self.shops_dict = {}
        for idx, node in enumerate(Centroids_Shop):
            self.shops_dict[idx] = node
        
        self.centroid_types_det_dict = {}
        self.centroid_types_det_dict_rev = {}
        for idx, node in enumerate(Centroid_Types_Detail):
            self.centroid_types_det_dict[idx] = node
            self.centroid_types_det_dict_rev[node] = idx

        self.centroid_types_RCh_dict = {}
        self.centroid_types_RCh_dict_rev = {}
        for idx, node in enumerate(Centroid_Types_RCh):
            self.centroid_types_RCh_dict[idx] = node
            self.centroid_types_RCh_dict_rev[node] = idx

        self.platforms_dict = {}
        self.platforms_dict_rev = {}
        for idx, node in enumerate(Platforms):
            self.platforms_dict[idx] = node
            self.platforms_dict_rev[node] = idx
                                    
        ''' ---------------------------------------------------
                Adds the vertices to the graph object
        ----------------------------------------------------'''
        
        self.G.add_nodes_from(Centroids_No_Platform, type='Centroids')
        self.G.add_nodes_from(self.ASE_measurement_nodes, type='ASE')
        self.G.add_nodes_from(Centroids_Platform, type='Platforms')
        self.G.add_nodes_from(self.VS_measurement_nodes, type='VS')
        self.G.add_nodes_from(self.other_nodes, type='Other')

        self.nodes = self.G.nodes()

        ''' ---------------------------------------------------
                Adds the edges to the graph object
        ----------------------------------------------------'''
        
        #We add the edges that are not ramps or escalators to the graph
        self.G.add_edges_from([('34d', '34w'), ('56d', '56w'), ('78d', '78w'), ('34w', '34c'), ('56w', '56c'), ('78w', '78c'), ('kioskh', 'KIOSK'), ('1w', '1d')], {'length': 3})
        self.G.add_edges_from([('34b', '34e'), ('56b', '56e'), ('78b', '78e'), ('34e', '34a'), ('56e', '56a'), ('78e', '78a')], {'length': 5})
        self.G.add_edges_from([('34b', '34e'), ('56b', '56e'), ('78b', '78e'), ('34e', '34a'), ('56e', '56a'), ('78e', '78a')], {'length': 5})
        self.G.add_edges_from([('34w', '56w'), ('34e', '56e')], {'length': 15.5})
        self.G.add_edges_from([('56w', '78w')], {'length': 14.4})
        self.G.add_edges_from([('56e', '56h'), ('56h', '78e'), ('neh2', 'neh1')], {'length': 7.2})
        self.G.add_edges_from([('34w', 'kioskh'), ('kioskh', '1w')], {'length': 11})
        self.G.add_edges_from([('1h2', '1h1'), ('1h1', '1D'), ('1c', '1h3'), ('1h3', '1h4'), ('1h4', '1C'), ('h', 'nh'), ('neh1', 'NE')], {'length': 2})  # Check length maybe
        self.G.add_edges_from([('1w', 'BAR'), ('nh', 'N')], {'length': 8})
        self.G.add_edges_from([('1h3', 'h')], {'length': 30})
        self.G.add_edges_from([('h', 'neh2')], {'length': 75})
        self.G.add_edges_from([('78C', '78B')], {'length': 82})
        self.G.add_edges_from([('neh2', '70fe')], {'length': 75})
        self.G.add_edges_from([('70fe', '70FE')], {'length': 10})
        self.G.add_edges_from([('1ab', '1AB')], {'length': 10})
        self.G.add_edges_from([('neh2', '1ab')], {'length': 40})
        self.G.add_edges_from([('nww', 'NW')], {'length': 60})
        self.G.add_edges_from([('nww', '1h2')], {'length': 10})
        self.G.add_edges_from([('34e', '1e'), ('nem', 'NEM')], {'length': 22})
        self.G.add_edges_from([('SW', 'sw'), ('sw', '9w')], {'length': 5})
        self.G.add_edges_from([('9w', '9h2'), ('9h2', '9h1'), ('9h1', '9d')], {'length': 3.5})
        self.G.add_edges_from([('9w', '9h3'), ('9h3', '9c')], {'length': 7})
        self.G.add_edges_from([('9w', '9h4'), ('9h4', '78w')], {'length': 8.6})
        self.G.add_edges_from([('9h1', 'SHOP'), ('nwm', 'NWM')], {'length': 6})
        self.G.add_edges_from([('SE', 'se'), ('se', '78e')], {'length': 5.5})
        self.G.add_edges_from([('1w', '1wc')], {'length': 3.0})
        self.G.add_edges_from([('nw', 'nwh'), ('nwm', 'nwh'), ('nwh', '1wh'), ('ne', '1eh'), ('nem', '1eh'), ('1eh', '1e')], {'length': 1.0})
        self.G.add_edges_from([('1wh', '1w')], {'length': 17.5})

        self.edge_labels_simple = dict([((u, v, ), d['length'])
                                        for u, v, d in self.G.edges(data=True)])
        
        self.edges = self.G.edges()
        
        #We add the edges reversed
        for edge in self.edges:
            self.G.add_edges_from([edge[::-1]], self.G.get_edge_data(edge[0], edge[1])[0])
            
        #We add the edges that are ramps or escalators and thus have a different weight in each direction
        self.G.add_edges_from([('34d', '34D'), ('56d', '56D'), ('78d', '78D'), ('9d', '9D'), ('34b', '34B'), ('56b', '56B'), ('78b', '78B'), ('56a', '56A'), ('1d', '1h2'), ('ne', 'neh2'), ('nw', 'NW')], {'length': 23.06})  # Stairs up
        self.G.add_edges_from([('34D', '34d'), ('56D', '56d'), ('78D', '78d'), ('9D', '9d'), ('34B', '34b'), ('56B', '56b'), ('78B', '78b'), ('56A', '56a'), ('1h2', '1d'), ('neh2', 'ne'), ('NW', 'nw')], {'length': 21.51})  # Stairs down
        self.G.add_edges_from([('34a', '34A'), ('78a', '78A')], {'length': 25.46})  # Stairs up plus some corridor
        self.G.add_edges_from([('34A', '34a'), ('78A', '78a')], {'length': 23.91})  # Stairs down plus some corridor
        self.G.add_edges_from([('34c', '34C'), ('56c', '56C'), ('78c', '78C'), ('9c', '9C')], {'length': 40.05})  # Ramps up
        self.G.add_edges_from([('34C', '34c'), ('56C', '56c'), ('78C', '78c'), ('9C', '9c')], {'length': 32.97})  # Ramps down
        self.G.add_edges_from([('1wc', '1c')], {'length': 33.8})  # Ramp up
        self.G.add_edges_from([('1c', '1wc')], {'length': 25.84})  # Ramp down

        self.edges = self.G.edges()

        self.edge_labels_duplicated = dict([((u, v, ), d['length'])
                                            for u, v, d in self.G.edges(data=True)])
                                                
        self.number_of_edges = len(self.edges)

        #We create a list of the edges associated to ASE sensors
        self.edges_ASE = [
            ('nw', 'nwh'),  # nwIn
            ('nwm', 'nwh'),  # nwmIn
            #('1c', '1wc') ,  # 1cIn MALFUNCTIONING
            ('nh', 'h'),  # nhIn
            ('neh1', 'neh2'),  # neh1In
            ('ne', '1eh'),  # neIn
            ('nem', '1eh'),  # nemIn
            ('sw', '9w'),  # swIn
            ('9h2', '9w'),  # gh2In
            ('9h3', '9w'),  # gh3In
            ('se', '78e'),  # seIn
            #('nww', '1h2') , #nwwIn MALFUNCTIONING
            
            ('nw', 'NW'),  # nwOut
            ('nwm', 'NWM'),  # nwmOut
            #('1c', '1h3'), #1cOut MALFUNCTIONING
            ('nh', 'N'),  # nhOut
            ('neh1', 'NE'),  # neh1Out
            ('ne', 'neh2'),  # neOut
            ('nem', 'NEM'),  # nemOut
            ('sw', 'SW'),  # swOut
            ('9h2', '9h1'),  # g9h2Out
            ('9h3', '9c'),  # 9h3Out
            ('se', 'SE'),  # seOut
            #('nww', 'NW') #nwwOut MALFUNCTIONING
        ]
        self.number_of_edges_ASE = len(self.edges_ASE)
        
        #We create a list of the edges associated to TINF
        self.edges_TINF = [
            ('1D', '1h1'),
            ('34d', '34w'),
            ('56d', '56w'),
            ('78d', '78w'),
            ('9d', '9h1'),
            ('1C', '1h4'),
            ('34c', '34w'),
            ('56c', '56w'),
            ('78c', '78w'),
            ('9c', '9h3'),
            ('1AB', '1ab'),
            ('70FE', '70fe'),
            ('34b', '34e'),
            ('56b', '56e'),
            ('78b', '78e'),
            ('34a', '34e'),
            ('56a', '56e'),
            ('78a', '78e'),
        ]
        
        self.edges_TINF_origins = ['1D', '34D', '56D', '78D', '9D', '1C', '34C', '56C', '78C', '9C', '1AB', '70FE', '34B', '56B', '78B', '34A', '56A', '78A']
        
        self.number_of_edges_TINF = len(self.edges_TINF)
        
        VS_inflow_edges = [('1wh', '1w'), ('1d', '1w'), ('1wc', '1w'), ('34d', '34w'), ('34c', '34w'), ('56d', '56w'), ('56c', '56w'), ('78d', '78w'), ('78c', '78w'), ('9h4', '78w'), ('1e', '34e'), ('34b', '34e'), ('34a', '34e'), ('56b', '56e'), ('56a', '56e'), ('56h', '56e')]
        VS_outflow_edges = [('1wh', 'nwh'), ('1d', '1h2'), ('1wc', '1c'), ('34d', '34D'), ('34c', '34C'), ('56d', '56D'), ('56c', '56C'), ('78d', '78D'), ('78c', '78C'), ('9h4', '9w'), ('1e', '1eh'), ('34b', '34B'), ('34a', '34A'), ('56b', '56B'), ('56a', '56A'), ('56h', '78e')]

#         #We create a list of the edges associated to historical data
#         self.edges_sales_points = [
#         ('KIOSK', 'kioskh'),
#         ('SHOP', '9h1'),
#         ('BAR', '1w'),
#         ]
#         self.number_of_edges_HIST = len(self.edges_sales_points)
        
        #We create dictionaries (for easier bookkeeping)
        self.edges_dict = {}
        self.edges_dict_rev = {}
        self.edges_reverse_dict = {}
        for idx, edge in enumerate(self.G.edges()):
            self.edges_reverse_dict[edge] = edge[::-1]
            self.edges_dict[idx] = edge
            self.edges_dict_rev[edge] = idx
       
        self.edges_ASE_dict = {}
        self.edges_ASE_dict_rev = {}
        for idx, edge in enumerate(self.edges_ASE):
            self.edges_ASE_dict[idx] = edge
            self.edges_ASE_dict_rev[edge] = idx
            
        self.edges_TINF_dict = {}
        self.edges_TINF_dict_rev = {}
        for idx, edge in enumerate(self.edges_TINF):
            self.edges_TINF_dict[idx] = edge
            self.edges_TINF_dict_rev[edge] = idx
            
        self.edges_TINF_origins_dict = {}
        #self.edges_TINF_origins_dict_rev = {}
        for idx, edge in enumerate(self.edges_TINF_origins):
            self.edges_TINF_origins_dict[idx] = edge
            #self.edges_TINF_origins_dict_rev[edge] = idx
#         self.edges_HIST_dict = {}
#         for idx, edge in enumerate(self.edges_sales_points):
#             self.edges_HIST_dict[idx] = edge

        self.VS_inflow_edges_dict = {}
        self.VS_inflow_edges_dict_rev = {}
        for idx, edge in enumerate(VS_inflow_edges):
            self.VS_inflow_edges_dict[idx] = edge
            self.VS_inflow_edges_dict_rev[edge] = idx
            
        self.VS_outflow_edges_dict = {}
        self.VS_outflow_edges_dict_rev = {}
        for idx, edge in enumerate(VS_outflow_edges):
            self.VS_outflow_edges_dict[idx] = edge
            self.VS_outflow_edges_dict_rev[edge] = idx

        # We define the areas. They are characterized by the nodes lying on their borders. They can overlap.
        areas = [['9h4', '78d', '56d', '34d', '1d', 'BAR', '1wh', '1wc', 'KIOSK', '34c', '56c', '78c'], ['56h', '56b', '34b', '1e', '34a', '56a']]
        
        #We create a dictionary of the areas
        self.areas_dict = {}
        for idx, area in enumerate(areas):
            self.areas_dict[idx] = area

        ## Code to generate all routes and write them in routes.txt (postprocessing needed to select the feasible routes)
        if estim_param.forceGenerateRoutes:
            routes = []
            path = nx.all_pairs_dijkstra_path(self.G, weight='length')
            for source in self.centroids:
                for target in self.centroids:
                    if source is not target:
                        routes.append(path[source][target])
            with open(estim_param.path_routes_file, 'w') as file:
                for item in routes:
                    file.write("{}\n".format(item))

        #We load the routes from a precoumpted file
        self.routes = []
        with open(estim_param.path_routes_file, 'r') as routesfile:
            for line in routesfile:
                line = line[1:-2]
                line = line.replace(',', '')
                line = line.replace("'", '')
                line = line.split()
                self.routes.append(line)
       
        self.number_of_routes = len(self.routes)

        # We generate all possible subroutes
        path = nx.all_pairs_dijkstra_path(self.G, weight='length')
         
        self.subroutes_west = []
        for source in VS_west_nodes:
            for target in VS_west_nodes:
                if source is not target:
                    self.subroutes_west.append(path[source][target])
                  
        self.subroutes_east = []
        for source in VS_east_nodes:
            for target in VS_east_nodes:
                if source is not target:
                    self.subroutes_east.append(path[source][target])
                            
        self.subroutes_VS = self.subroutes_west + self.subroutes_east
       
        #We create dictonaries for the routes and subroutes
        self.routes_dict = {}
        self.routes_dict_rev = {}

        for idx, route in enumerate(self.routes):
            self.routes_dict[idx] = route
            self.routes_dict_rev[str(route)] = idx

        self.subroutes_VS_dict = {}
        self.subroutes_VS_dict_rev = {}
        for idx, route in enumerate(self.subroutes_VS):
            self.subroutes_VS_dict[idx] = route
            self.subroutes_VS_dict_rev[str(route)] = idx
            
        self.VS_nodes_dict = {}
        self.VS_nodes_dict_rev = {}
        for idx, node in enumerate(VS_nodes):
            self.VS_nodes_dict[idx] = node
            self.VS_nodes_dict_rev[node] = idx

        # We compute the distances between edges and routes
        self.distances_edge_route = np.zeros((len(self.edges), len(self.routes)))
  
        for e in range(len(self.edges)):
            for r in range(len(self.routes)):
                # if the edge in on the route, we compute the distance between the start of the route and the start of the edge
                if '( ' + str(self.edges[e][0]) + ',' + str(self.edges[e][1]) + ')' in ['(' + str(self.routes[r][i - 1]) + ',' + str(self.routes[r][i]) + ')' for i in range(1, len(self.routes[r]))]:
                    self.distances_edge_route[e][r] = nx.shortest_path_length(self.G, source=self.routes[r][0], target=self.edges[e][0], weight='length')
                else:
                    self.distances_edge_route[e][r] = -1
  
        self.distances_edge_route2 = np.zeros((len(self.edges_ASE), len(self.routes)))
        
        # Exit links
        self.exit_centroids_exit_edges = {
            'SW': ('sw', 'SW'),
            'NW': ('nw', 'NW'),
            'NWM': ('nwm', 'NWM'),
            'SE': ('se', 'SE'),
            'NEM': ('nem', 'NEM'),
            'N': ('nh', 'N'),
            'NE': ('neh1', 'NE')
            #'NW': ('nww', 'NW'), negligibly small
        }
        
        # dictionaries for route choice
        
        self.route_orig_dict = {}
        self.route_dest_dict = {}
        
        self.routes_from_centroid = defaultdict(list)
        self.routes_to_centroid = defaultdict(list)
        
        for route_key in self.routes_dict.keys():
            orig_centroid_key = self.centroids_dict_rev[self.routes_dict[route_key][0]]
            dest_centroid_key = self.centroids_dict_rev[self.routes_dict[route_key][-1]]
            self.route_orig_dict[route_key] = orig_centroid_key
            self.route_dest_dict[route_key] = dest_centroid_key
        
            self.routes_from_centroid[orig_centroid_key].append(route_key)
            self.routes_to_centroid[dest_centroid_key].append(route_key)

        # dictionaries for postprocessing of VisioSafe
        
        self.subroute_orig_dict = {}
        self.subroute_dest_dict = {}
        
        self.subroutes_from_node = defaultdict(list)
        self.subroutes_to_node = defaultdict(list)
        
        for subroute_key in self.subroutes_VS_dict.keys():
            orig_centroid_key = self.VS_nodes_dict_rev[self.subroutes_VS_dict[subroute_key][0]]
            dest_centroid_key = self.VS_nodes_dict_rev[self.subroutes_VS_dict[subroute_key][-1]]
            self.subroute_orig_dict[subroute_key] = orig_centroid_key
            self.subroute_dest_dict[subroute_key] = dest_centroid_key
        
            self.subroutes_from_node[orig_centroid_key].append(subroute_key)
            self.subroutes_to_node[dest_centroid_key].append(subroute_key)

        # linking track numbres to platform names
        self.track_platform_dict = {
            1: '1',
            3: '3/4',
            4: '3/4',
            5: '5/6',
            6: '5/6',
            7: '7/8',
            8: '7/8',
            9: '9',
            70: '70'}

        self.centroid_platform_dict = {
            '1D': '1',
            '1C': '1',
            '70FE': '70',
            '1AB': '1',
            '34D': '3/4',
            '34C': '3/4',
            '34B': '3/4',
            '34A': '3/4',
            '56D': '5/6',
            '56C': '5/6',
            '56B': '5/6',
            '56A': '5/6',
            '78D': '7/8',
            '78C': '7/8',
            '78B': '7/8',
            '78A': '7/8',
            '9D': '9',
            '9C': '9'
        }
        
        # Structural components of Lausanne useful for Circos and other highly aggregated plots
        self.structural_labels = ['P1', 'P34', 'P56', 'P78', 'P9', 'P70', 'Metro', 'North', 'South', 'Shops', ]
        #Dictionary that aggregates the centroids in these structural zones:
        self.structural_centroids_dict = {'1D': 0, '1C': 0, '70FE': 5, '1AB': 0, '34D': 1, '34C': 1, '34B': 1, '34A': 1, '56D': 2, '56C': 2, '56B': 2, '56A': 2, '78D': 3, '78C': 3, '78B': 3, '78A': 3, '9D': 4, '9C': 4, 'NW': 7, 'NWM': 6, 'NE': 7, 'NEM': 6, 'SW': 8, 'SE': 8, 'N': 7, 'KIOSK': 9, 'BAR': 9, 'SHOP': 9}
        
        self.ASE_edge_names_dict = {'ASE9ab_in': ('9h3', '9w'),
                                    'ASE9cde_in': ('sw', '9w'),
                                    'ASE9fgh_in': ('9h2', '9w'),
                                    'ASE4_out': ('1c', '1wc'),
                                    'ASE5a_in': ('nw', 'nwh'),
                                    'ASE10_in': ('nwm', 'nwh'),
                                    'ASE8_in': ('se', '78e'),
                                    'ASE2_out': ('ne', '1eh'),
                                    'ASE2de_out': ('nem', '1eh'),
                                    'ASE3_in': ('nh', 'h'),
                                    'ASE1_in': ('neh1', 'neh2'),
                                    'ASE6_in': ('nww', '1h2'),
                                    
                                    'ASE9ab_out': ('9h3', '9c'),
                                    'ASE9cde_out': ('sw', 'SW'),
                                    'ASE9fgh_out': ('9h2', '9h1'),
                                    'ASE4_in': ('1c', '1h3'),
                                    'ASE5a_out': ('nw', 'NW'),
                                    'ASE10_out': ('nwm', 'NWM'),
                                    'ASE8_out': ('se', 'SE'),
                                    'ASE2_in': ('ne', 'neh2'),
                                    'ASE2de_in': ('nem', 'NEM'),
                                    'ASE3_out': ('nh', 'N'),
                                    'ASE1_out': ('neh1', 'NE'),
                                    'ASE6_out': ('nww', 'NW'),
                                    }
    
        #Reversed dictionary:
        self.ASE_edge_names_dict_rev = {}
        for key in self.ASE_edge_names_dict.keys():
            self.ASE_edge_names_dict_rev[self.ASE_edge_names_dict[key]] = key
            
        self.edges_sens_correction = [
            ('nw', 'nwh'),  # nwIn
            ('nwm', 'nwh'),  # nwmIn
            ('nh', 'h'),  # nhIn
            ('neh1', 'neh2'),  # neh1In
            
            ('nw', 'NW'),  # nwOut
            ('nwm', 'NWM'),  # nwmOut
            ('nh', 'N'),  # nhOut
            ('neh1', 'NE'),  # neh1Out
            
            # following links tentatively considered
            ('sw', '9w'),  # swIn
            ('9h2', '9w'),  # gh2In
            ('9h3', '9w'),  # gh3In
            ('se', '78e'),  # seIn
            
            ('sw', 'SW'),  # swOut
            ('9h2', '9h1'),  # g9h2Out
            ('9h3', '9c'),  # 9h3Out
            ('se', 'SE'),  # seOut
        ]
        
    def problem_size(self, estim_param):
        '''
        Based on the weights defined for the problem,
        calculates the number of active data sets, hence constraints
        '''
        self.no_constraints = 0
        if estim_param.muZero > 0:
            self.no_constraints += len(self.routes_dict) * len(estim_param.tint_dict)
        if estim_param.muLink > 0:
            self.no_constraints += len(self.edges_ASE_dict) * len(estim_param.tint_dict)
        if estim_param.muSubRoute > 0:
            self.no_constraints += len(self.subroutes_VS_dict) * len(estim_param.tint_dict)
        if estim_param.muTINF > 0:
            self.no_constraints += len(self.edges_TINF_dict) * len(estim_param.tint_dict)
        if estim_param.muHist > 0:
            self.no_constraints += len(self.routes_dict) * len(estim_param.tint_dict)  # Approximation
        if estim_param.muHist > 0:
            self.no_constraints += 2 * len(self.shops_dict) + len(self.centroids_np_dict) + len(self.centroids_dict)
            
        self.no_variables = len(self.routes_dict) * len(estim_param.tint_dict)
