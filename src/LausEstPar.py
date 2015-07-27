'''
@author: flurin, nicholas
'''
import datetime
import time
#import numpy as np
import multiprocessing


class LausEstimParam:
    
    # Operating system specification, useful for multi-processing
    good_op_sys = True   # Set to True if on UNIX/Mac/Linux, to false if on Windows
    
    if good_op_sys is False:
        print "Operating system is set to Windows"

    date_format = "%d-%b-%Y %H:%M:%S"
    time_format = "%H:%M:%S"
    delta_t = 60  # time interval, in seconds
    
    # Maximum travel time
    max_TT_int = 10  # Should be set to >=10!
    
    # Pedestrian walking speed distribution
    vel_mean = 1.34
    vel_stdev = 0.34
    
    #input and output directories
    clearOutput = True
    
    # parameters of TINF framework
    semipath_FQ = 'TTT/example_data_set_train_changeover.csv'
    semipath_OTT = 'TTT/example_data_set_effective_timetable.csv'
    semipath_TINFParamDist = 'TTT/example_data_set_TINFParamDist.csv'
    semipath_alphaModel = 'TTT/example_data_set_AlphaModelFit.csv'
    semipath_alphaModelRed = 'TTT/example_data_set_AlphaModelFitRed.csv'
    FQ_std = 0.2  # Relative standard deviation of FQ counts

    # specification of routes
    semipath_routes_file = 'routes/example_data_set_routes.txt'
    
    # pre-computed assignment
    forceRecompAssgnment = False
    forceRecompDemand = False
    forceGenerateRoutes = False  # Careful, if TRUE overwrites manual route map!
    parallel_threads = multiprocessing.cpu_count()

    semipath_precomp = "precomputed/"
    semipath_output_plots = "plot_output/"
    semipath_output_text = "text_output/"
    
    # ASE sensor correction
    sens_corr_par_a = 1.0
    sens_corr_par_b = 0.0

    # Visualization options (set to True or False)
    show_network_structure = True
    show_dynamic_flow_map = True
    show_dynamic_OD_matrix = True
    create_circos = True
    show_subroute_demand = True
    show_total_demand = True
    show_occupation = True
    compare_ASE_VS = True
    compare_TINF_VS = True
    show_TINF = True
    show_ASE = True
    show_VS_flows = True
    show_scatter_plots = True
    
    plot_show = False
    file_format = '.png'  # e.g. .eps or .png
    LaTeX = True
    
    axis_ticks_spacing = 3
    
    def __init__(self, date_var, start_time, end_time, start_time_eval, end_time_eval,
                 wZero, wASE, wVS, wTINF, wHist, wHistAgg,
                 iter_MC, corr_ASE_counts, case_name, path_input, semipath_output):
        '''
        Constructor for this class. Sets th various parameters into the class
        to be used later by various classes or functions.
        '''
        self.case_study_name = case_name  # + "_" + self.date
        
        print "Loading estimation parameters of " + case_name
        
        if isinstance(date_var, str):
            self.date = date_var
            self.date_set = None
        elif isinstance(date_var, list):
            self.date = "09-Feb-1967"
            self.date_set = date_var
        
        self.start_date = self.date + " " + start_time
        self.end_date = self.date + " " + end_time

        self.start_date_eval = self.date + " " + start_time_eval
        self.end_date_eval = self.date + " " + end_time_eval
        
        self.start_date_t = datetime.datetime.strptime(self.start_date, self.date_format)
        self.end_date_t = datetime.datetime.strptime(self.end_date, self.date_format)
        self.num_tint = int((self.end_date_t - self.start_date_t).total_seconds() / self.delta_t)
        
        self.start_date_t_eval = datetime.datetime.strptime(self.start_date_eval, self.date_format)
        self.end_date_t_eval = datetime.datetime.strptime(self.end_date_eval, self.date_format)
        self.num_tint_eval = int((self.end_date_t_eval - self.start_date_t_eval).total_seconds() / self.delta_t)
        
        # Weights in NNLS
        self.muZero = wZero
        self.muLink = wASE
        self.muSubRoute = wVS
        self.muTINF = wTINF
        self.muHist = wHist
        self.muHistAgg = wHistAgg
        
        # Number of iterations in MC integration (should be large)
        self.MC_iterations = iter_MC  # Set to 1 if deterministic framework
        self.correct_ASE_counts = corr_ASE_counts
        
        self.runtime_helper = time.time()
        self.runtime_start = time.time()
        
        # generate dynamic paths
        self.path_FQ = path_input + self.semipath_FQ
        self.path_OTT = path_input + self.semipath_OTT
        self.path_TINFParamDist = path_input + self.semipath_TINFParamDist
        self.path_alphaModel = path_input + self.semipath_alphaModel
        self.path_alphaModelRed = path_input + self.semipath_alphaModelRed
        
        # parameters of link flow and subroute flow data
        ase_date = str(self.start_date_t.year) + "_" + str(self.start_date_t.month).zfill(2) + "_" + str(self.start_date_t.day).zfill(2)
        vs_date = str(self.start_date_t.year) + "-" + str(self.start_date_t.month).zfill(2) + "-" + str(self.start_date_t.day).zfill(2)
        self.semipath_ASE_LS_file = 'ASE/' + ase_date + '_LS.csv'
        self.semipath_ASE_add_file = 'ASE/' + ase_date + '_add.csv'
        self.semipath_VisioSafe_file = 'VisioSafe/al_position' + vs_date + 'CFFLausanne_disagg_od.csv'
            
        self.path_ASE_LS_file = path_input + self.semipath_ASE_LS_file
        self.path_ASE_add_file = path_input + self.semipath_ASE_add_file
        self.path_VisioSafe_file = path_input + self.semipath_VisioSafe_file
        
        self.path_routes_file = path_input + self.semipath_routes_file
        
        self.path_output = semipath_output + '_' + case_name + '/'
        self.path_precomp = path_input + self.semipath_precomp
        self.path_output_plots = self.path_output + self.semipath_output_plots
        self.path_output_text = self.path_output + self.semipath_output_text
        
        time_span_min = (self.end_date_t - self.start_date_t).seconds / 60
        self.namePrecompAssgnment = 'assg_' + str(time_span_min) + 'min'
        
        # Creates dict of time intervals
        self.tint_dict = {}
        self.tint_dict_rev = {}
        for intervals_elapsed in range(0, self.num_tint):
            curr_dat = self.start_date_t + datetime.timedelta(seconds=self.delta_t * intervals_elapsed)
            self.tint_dict[intervals_elapsed] = curr_dat.strftime(self.date_format)
            self.tint_dict_rev[curr_dat.strftime(self.date_format)] = intervals_elapsed
            
        # Creates dict of time intervals for evaluation
        self.tint_eval_dict = {}
        self.tint_eval_dict_rev = {}
        for intervals_elapsed in range(0, self.num_tint_eval):
            curr_dat = self.start_date_t_eval + datetime.timedelta(seconds=self.delta_t * intervals_elapsed)
            self.tint_eval_dict[intervals_elapsed] = curr_dat.strftime(self.date_format)
            self.tint_eval_dict_rev[curr_dat.strftime(self.date_format)] = intervals_elapsed
        
        self.tint_eval_start = self.tint_dict_rev[self.tint_eval_dict[0]]

        self.tint_full2eval_dict = {}
        self.tint_eval2full_dict = {}
        for tint in self.tint_eval_dict.keys():
            self.tint_full2eval_dict[tint + self.tint_eval_start] = tint
            self.tint_eval2full_dict[tint] = tint + self.tint_eval_start
                
        self.path_precomputed_asg = self.path_precomp + self.namePrecompAssgnment
        self.path_precomputed_dem = self.path_precomp + self.case_study_name
    
        self.path_plots = self.path_output_plots + self.case_study_name + "_"
        self.path_textout = self.path_output_text + self.case_study_name + "_"
                
        self.print_incr_runtime()
            
    def print_incr_runtime(self, opt_mess=None, time_elaps='incr', returnString=False):
        '''
        This method calculates then prints the running time from the previous
        call to this function. It can return the value as a string.
        '''
        if time_elaps is 'incr':
            delta_t = time.time() - self.runtime_helper
        elif time_elaps is 'full':
            delta_t = time.time() - self.runtime_start
        else:
            delta_t = -99
            print "An error has occurred."
        
        printMess = "(completed in "
        printMess += str("%.2f" % delta_t)
        if opt_mess is None:
            printMess += "s)"
        else:
            printMess += "s," + opt_mess + ")"
            
        if returnString:
            return printMess
        else:
            self.runtime_helper = time.time()
            print printMess
