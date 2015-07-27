'''
Created on Jul 11, 2014

@author: flurin
'''
from LausEstPar import LausEstimParam
from LausNetw import LausNetwork
from LausSensDat import LausSensData
from Assg import Assignment
from Prior import DemPrior
from DemEst import DemandEstimation
from Eval import Evaluation
from EvalMD import EvaluationMD
from PostProc import PostProcessing


class DemandEstimator:
    '''
    This class contains the calls to the important class and methods
    of the simulation framework. This class contains different parts
    depending on which data is available. This class contains only
    a constructor.
    Depending on the date_set argument, the framework will switch to a
    single day prediction or a multi-day prediction. The following classes are
    used mutliple time:
    - LausEstimParam: packs all the parameters, settings and paths into one class
    - LausNetwork: Specification of the network (graph)
    - Assignment: Calculates the assignment matrices, or loads them if they are already calculated
    - LausSensData: Class containing the sensor data
    - Prior: Creates train induced flows based on train time tables
    - DemandEstimation: Actually performs the estimation of the flow in the station
    - Evaluation: Processes the data from the DemandEstimation
    - PostProcessing: Creates the visualization
    '''
    def __init__(self, date_set, start_time, end_time, start_time_eval, end_time_eval,
                 wZero, wASE, wVS, wTINF, wHist, wHistAgg, iter_MC, corr_ASE, case_name, path_input, semipath_output,
                 wZero_ref=None, wASE_ref=None, wVS_ref=None, wTINF_ref=None, wHist_ref=None, wHistAgg_ref=None,
                 iter_MC_ref=None, corr_ASE_ref=None, case_name_ref=None):
        '''
        Constructor of this class which actually calls the main calculations.
        '''

        #---------------- For one single day --------------#
        if isinstance(date_set, str):
            
            date = date_set
            
            # Class containing all the parameters for the case, with the paths for saving results
            estim_param = LausEstimParam(date, start_time, end_time, start_time_eval, end_time_eval,
                                         wZero, wASE, wVS, wTINF, wHist, wHistAgg,
                                         iter_MC, corr_ASE, case_name, path_input, semipath_output)
            
            # Loads the network
            network = LausNetwork(estim_param)
            
            # Computes assignment mapping
            assignment = Assignment(estim_param, network)
            
            # Loads the sens_data from count/tracking sensors.
            sens_data = LausSensData(estim_param, network)
            
            # Compute prior
            prior = DemPrior(estim_param, network, sens_data)
                       
            # Compute dem_est (MC integration)
            estimate = DemandEstimation(estim_param, network, assignment, sens_data, prior)
              
            # Evaluate dem_est (post-processing)
            evaluate = Evaluation(estim_param, network, assignment, sens_data, prior, estimate)
              
            # No simulation, only data to visualize
            if case_name_ref is None:
                PostProcessing(estim_param, network, evaluate)
                
            # simulates based on the reference case
            else:
                estim_param_ref = LausEstimParam(date, start_time, end_time, start_time_eval, end_time_eval,
                                                 wZero_ref, wASE_ref, wVS_ref, wTINF_ref, wHist_ref, wHistAgg_ref,
                                                 iter_MC_ref, corr_ASE_ref, case_name_ref, path_input, semipath_output)
                prior_ref = DemPrior(estim_param_ref, network, sens_data)
                estimate_ref = DemandEstimation(estim_param_ref, network, assignment, sens_data, prior_ref)
                evaluate_ref = Evaluation(estim_param_ref, network, assignment, sens_data, prior_ref, estimate_ref)
                PostProcessing(estim_param, network, evaluate, estpar_ref=estim_param_ref, eval_ref=evaluate_ref)
                
        #---------------- For multiple days ---------------#
        elif isinstance(date_set, list):
            
            cur_case = case_name + '_set'
            estim_param_base = LausEstimParam(date_set, start_time, end_time, start_time_eval, end_time_eval,
                                              wZero, wASE, wVS, wTINF, wHist, wHistAgg,
                                              iter_MC, corr_ASE, cur_case, path_input, semipath_output)
            eval_set = {}
            
            if not case_name_ref is None:
                cur_case_ref = case_name_ref + '_set_ref'
                estim_param_base_ref = LausEstimParam(date_set, start_time, end_time, start_time_eval, end_time_eval,
                                                      wZero_ref, wASE_ref, wVS_ref, wTINF_ref, wHist_ref, wHistAgg_ref,
                                                      iter_MC_ref, corr_ASE_ref, cur_case_ref, path_input, semipath_output)
                eval_set_ref = {}
            
            network = LausNetwork(estim_param_base)
            assignment = Assignment(estim_param_base, network)
            
            for date in date_set:
                
                cur_case = case_name + '_' + date
                
                estim_param = LausEstimParam(date, start_time, end_time, start_time_eval, end_time_eval,
                                             wZero, wASE, wVS, wTINF, wHist, wHistAgg,
                                             iter_MC, corr_ASE, cur_case, path_input, semipath_output)
                sens_data = LausSensData(estim_param, network)
                prior = DemPrior(estim_param, network, sens_data)
                estimate = DemandEstimation(estim_param, network, assignment, sens_data, prior)
                evaluate = Evaluation(estim_param, network, assignment, sens_data, prior, estimate)
                eval_set[date] = evaluate
                   
                if not case_name_ref is None:
                    cur_case_ref = case_name_ref + '_' + date
                    
                    estim_param_ref = LausEstimParam(date, start_time, end_time, start_time_eval, end_time_eval,
                                                     wZero_ref, wASE_ref, wVS_ref, wTINF_ref, wHist_ref, wHistAgg_ref,
                                                     iter_MC_ref, corr_ASE_ref, cur_case_ref, path_input, semipath_output)
                    prior_ref = DemPrior(estim_param_ref, network, sens_data)
                    estimate_ref = DemandEstimation(estim_param_ref, network, assignment, sens_data, prior_ref)
                    evaluate_ref = Evaluation(estim_param_ref, network, assignment, sens_data, prior_ref, estimate_ref)
                    eval_set_ref[date] = evaluate_ref
                    
            evaluateMD = EvaluationMD(estim_param_base, network, eval_set)
            
            ## Evaluate dem_est (post-processing)
            if case_name_ref is None:
                PostProcessing(estim_param_base, network, evaluateMD)
            else:
                evaluateMD_ref = EvaluationMD(estim_param_base_ref, network, eval_set_ref)
                PostProcessing(estim_param_base, network, evaluateMD, estim_param_base_ref, evaluateMD_ref)
