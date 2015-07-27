'''
@author: flurin, Eduard
'''

import datetime
import pandas as pd
from pandas import DataFrame


class LausSensData:
    '''
    Loads the various sensor data for the station in Lausanne.
    The data sources are link flow counts from ASE, some sales data
    which is similar to link flow counts and tracking data from VisioSafe.
    '''
    def __init__(self, estim_param, network):
        '''
        Constructor which calls the methods defined in the class
        '''
        print "Loading Lausanne sensor data",

        '''------------------------------------------------------------------
              Initializes the variables and sets the paths to the files
        ------------------------------------------------------------------'''
        self.network = network
        self.estim_param = estim_param
        self.path_ASE_LS_file = estim_param.path_ASE_LS_file
        self.path_ASE_add_file = estim_param.path_ASE_add_file
        self.path_VisioSafe_file = estim_param.path_VisioSafe_file

        self.f_hat = []
        self.g_hat = []
        self.n_hat = []
        
        self.GdL_subroute_flows()
        self.GdL_link_flow_counts(estim_param, network)
        self.GdL_accumulation_flows()
        
        # sales data (per minute) -- a priori assumptions based on monthly sales
        self.sales_per_minute = {'KIOSK': 1.0,
                                 'BAR': 1.0,
                                 'SHOP': 1.0}
    
        # fractions of platform departure flows by platform sector -- a priori assumption based on Molyneaux, 2014
        sector_dest_flow_fractions = {'A': 0.095,
                                      'B': 0.271,
                                      'C': 0.475,
                                      'D': 0.159}
    
        self.centroid_platform_dest_flow_fractions = {'1D': 0.25,
                                                      '1C': 0.5,
                                                      '70FE': 1.0,
                                                      '1AB': 0.25,
                                                      '34D': sector_dest_flow_fractions['D'],
                                                      '34C': sector_dest_flow_fractions['C'],
                                                      '34B': sector_dest_flow_fractions['B'],
                                                      '34A': sector_dest_flow_fractions['A'],
                                                      '56D': sector_dest_flow_fractions['D'],
                                                      '56C': sector_dest_flow_fractions['C'],
                                                      '56B': sector_dest_flow_fractions['B'],
                                                      '56A': sector_dest_flow_fractions['A'],
                                                      '78D': sector_dest_flow_fractions['D'],
                                                      '78C': sector_dest_flow_fractions['C'],
                                                      '78B': sector_dest_flow_fractions['B'],
                                                      '78A': sector_dest_flow_fractions['A'],
                                                      '9D': 0.75,
                                                      '9C': 0.25}
        
        self.beta_p2np = 0.9135
        self.beta_np2p = 0.95
        
        estim_param.print_incr_runtime()
             
    def GdL_link_flow_counts(self, estim_param, network):
        '''
        Sets the observed link flow counts into f_hat
        '''
        ASE_LS = self.ASEdataPreprocessing(network)
        
        #We create the vector of the sensors flows:
        for t in self.estim_param.tint_dict.keys():
            t_int = self.estim_param.tint_dict[t]
            t_int = datetime.datetime.strptime(t_int, self.estim_param.date_format)
            for ASE_edge_key in self.network.edges_ASE_dict.keys():
                count_obs = ASE_LS.ix[t_int, self.network.edges_ASE_dict[ASE_edge_key]]
                
                if estim_param.correct_ASE_counts:
                    if network.edges_ASE_dict[ASE_edge_key] in network.edges_sens_correction:
                        count_obs = self.sensor_correction_ASE(estim_param, count_obs)
                
                self.f_hat.append(count_obs)
            
    def GdL_subroute_flows(self):
        '''
        Sets the link flow counts on the subroutes into g_hat
        '''
        disaggOD = self.VSdataPreprocessing()
        disaggOD['time_origin'] = disaggOD['time_origin'].apply(lambda x: x.replace(second=0, microsecond=0))
        #We create the vector of the subroute flows:
        for tint in self.estim_param.tint_dict.values():
            tint = datetime.datetime.strptime(tint, self.estim_param.date_format)
            for p in self.network.subroutes_VS_dict.values():
                self.g_hat.append(len(disaggOD[(disaggOD.time_origin == tint) & (disaggOD.origin == p[0]) & (disaggOD.destination == p[-1])]))
            
    def GdL_accumulation_flows(self):
        '''
        Sets the accumulation flows into accumulation_hat
        '''
        disaggOD = self.VSdataPreprocessing()
                
        for tint in self.estim_param.tint_dict.values():
            tint = datetime.datetime.strptime(tint, self.estim_param.date_format)
            for area in self.network.areas_dict.values():
                n = 0
                for idx, person in disaggOD.iterrows():
                    if person['origin'] in area:
                        n = n + self.calculate_presence_time(person['time_origin'], person['time_destination'], tint) / float(self.estim_param.delta_t)
                self.n_hat.append(n)
 
    def calculate_presence_time(self, t_start, t_end, tint):
        '''
        Returns the time spent in a given interval
        '''
        t_in = tint
        t_out = tint + datetime.timedelta(seconds=self.estim_param.delta_t)
        
        if t_start <= t_in <= t_end <= t_out:
            return (t_end - t_in).total_seconds()
        elif t_in <= t_start <= t_out <= t_end:
            return (t_out - t_start).total_seconds()
        elif t_in <= t_start <= t_end <= t_out:
            return (t_end - t_start).total_seconds()
        elif t_start <= t_in <= t_out <= t_end:
            return (t_out - t_in).total_seconds()
        else:
            return 0

    def matlab2datetime(self, matlab_datenum):
        '''
        Transforms matlab time to python's datetime:
        '''
        day = datetime.datetime.fromordinal(int(matlab_datenum))
        dayfrac = datetime.timedelta(days=matlab_datenum % 1) - datetime.timedelta(days=366)
        return day + dayfrac
         
    def stringToDateTime(self, string):
        '''
        Transforms a string to date
        '''
        return datetime.datetime.strptime(string, '%d-%m-%Y %H:%M:%S')
        
    def ASEdataPreprocessing(self, network):
        '''
        Loads then processes the data from ASE. The data is processed to make it easily
        usable latter on. Basically so it has the same format as the rest of the framework.
        '''
        #************** ASE Data **************
        ASE_LS = pd.read_csv(self.path_ASE_LS_file, dtype=int)
        ASE_add = pd.read_csv(self.path_ASE_add_file)
        
        #Table to rename the edges from the file to the names used in LausNetw
        self.ASE_edges_conversion = network.ASE_edge_names_dict
        
        #Edges that we need from ASE.add file:
        edges_from_ASE_add = ('ASE1_in', 'ASE1_out', 'ASE3_in', 'ASE3_out', 'ASE6_in', 'ASE6_out')
        
        #We take the edges that we need and add them in ASE_LS
        ASE_add2 = DataFrame(ASE_add.ix[:, edges_from_ASE_add])
        ASE_LS = ASE_LS.add(ASE_add2, fill_value=0)
        
        #We merge the counts of the sensors (8a,8b) and (2ab, 2c)
        ASE_LS['ASE8_in'] = ASE_LS['ASE8a_in'] + ASE_LS['ASE8b_in']
        ASE_LS['ASE8_out'] = ASE_LS['ASE8a_out'] + ASE_LS['ASE8b_out']
        ASE_LS['ASE2_in'] = ASE_LS['ASE2ab_in'] + ASE_LS['ASE2c_in']
        ASE_LS['ASE2_out'] = ASE_LS['ASE2ab_out'] + ASE_LS['ASE2c_out']
                 
        #We delete the counts that have been merged
        del ASE_LS['ASE8a_in']
        del ASE_LS['ASE8a_out']
        del ASE_LS['ASE8b_in']
        del ASE_LS['ASE8b_out']
        del ASE_LS['ASE2ab_in']
        del ASE_LS['ASE2ab_out']
        del ASE_LS['ASE2c_in']
        del ASE_LS['ASE2c_out']
            
        #We rename the edges
        ASE_LS = ASE_LS.rename(columns=self.ASE_edges_conversion)
        
        ASE_LS['start_date'] = ASE_LS.start_day.astype(int).map(str) + '-' + ASE_LS.start_month.astype(int).map(str) + '-' + ASE_LS.start_year.astype(int).map(str) + ' ' + ASE_LS.start_hour.astype(int).map(str) + ':' + ASE_LS.start_minute.astype(int).map(str) + ':' + ASE_LS.start_second.astype(int).map(str)
        ASE_LS['start_date'] = ASE_LS['start_date'].apply(self.stringToDateTime)
        ASE_LS = ASE_LS.set_index('start_date', drop=True)
        return ASE_LS

    #Processes and transforms the data from the VisioSafe sensors
    def VSdataPreprocessing(self):
        '''
        Loads and processes the data from VisioSafe. The data is processed to make it easily
        usable latter on. Basically so it has the same format as the rest of the framework.
        '''
        
        #************** VisioSafe Data **************
        disaggOD = pd.read_csv(self.path_VisioSafe_file, header=None)

        #We add the name of each column of the file
        disaggOD.columns = ['ped_id', 'origin', 'destination', 'dist', 'time', 'speed', 'time_origin', 'time_destination', 'route']

        #Dictionary that maps origin/destination names to nodes of LausNetwork
        VStoNodes = {1: '9h4',
                     2: '9h4',
                     3: '9h4',
                     4: '9h4',
                     -201: '9h4',
                     5: '78d',
                     6: '78c',
                     7: '56d',
                     8: '56c',
                     9: '34d',
                     10: '34c',
                     11: '1d',
                     12: '1wc',
                     13: '1wh',
                     14: '1wh',
                     13.5: '1wh',
                     -214: '1wh',
                     -215: '1wh',
                     -301: '56h',
                     -302: '56h',
                     -303: '56h',
                     17: '56b',
                     18: '56a',
                     19: '34b',
                     20: '34a',
                     -311: '1e',
                     -312: '1e',
                     -313: '1e',
                     21: '1e',
                     22: '1e',
                     23: '1e'}
        
        #We keep the data we are interested in
        disaggOD = disaggOD[['origin', 'destination', 'time_origin', 'time_destination']]
        
        #We apply the conversion of the origin/destination nodes using the VStoNodes dictionary
        disaggOD['origin'] = disaggOD.origin.apply(lambda x: VStoNodes.get(x, 'Nan'))
        disaggOD['destination'] = disaggOD.destination.apply(lambda x: VStoNodes.get(x, 'Nan'))
        
        #Filtering of the measures that don't fulfil the specifications (MAYBE MORE TO BE ADDED)
        disaggOD = disaggOD[disaggOD.origin != 'Nan']
        disaggOD = disaggOD[disaggOD.destination != 'Nan']
        disaggOD = disaggOD[(disaggOD.time_origin > 1) & (disaggOD.time_destination > 1) & (disaggOD.origin != disaggOD.destination)]
        
        #Transformation from Matlab date format to Python date format
        disaggOD['time_origin'] = disaggOD.time_origin.apply(self.matlab2datetime)
        disaggOD['time_destination'] = disaggOD.time_destination.apply(self.matlab2datetime)
        
        return disaggOD
    
    def sensor_correction_ASE(self, estim_param, count_obs):
        '''
        Modifies the way the ASE sensors counts. To deal with bidirectionality.
        '''
        return estim_param.sens_corr_par_a * (count_obs ** 2) + estim_param.sens_corr_par_b * count_obs
