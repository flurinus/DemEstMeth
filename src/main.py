'''
Created on Mar 19, 2014
@author: flurin, nicholas, quentin, eduard
'''

#import sys
#sys.path.append('dempy')
from DemEstimator import DemandEstimator


def main():
    '''
    Definition of the main function with the required parameters
    to run the simulation.
    '''
    
    # Date & time for estimation
    date_set = '22-Jan-2013'
    start_time = "07:30:00"
    end_time = "07:59:00"
    
    # Date & time for evaluation
    start_time_eval = "07:37:00"
    end_time_eval = "07:52:00"
    
    # Reference case
    case_name_ref = "RefNoZero"

    # Weights
    wZero_ref = 0
    wASE_ref = 1
    wVS_ref = 0
    wTINF_ref = 0
    wHist_ref = 0
    wHistAgg_ref = 0
    
    iter_MC_ref = 1
    corr_ASE_ref = True
    
    # Environmental variables
    path_input = '../input_data_rand/'
    semipath_output = '../output_rand/'
    
    # Current case:
    case_name = "THCNoZero"

    # Weights
    wZero = 0

    wASE = 1
    wVS = 0
    wTINF = 0.75
    wHist = 0
    wHistAgg = 1e-1
    
    iter_MC = 4
    corr_ASE = True

    # Individual day analysis
    #for date in date_set:
    date = date_set
    cur_case_name = case_name + '_' + date
    cur_case_name_ref = case_name_ref + '_' + date
    DemandEstimator(date, start_time, end_time, start_time_eval, end_time_eval,
                    wZero, wASE, wVS, wTINF, wHist, wHistAgg, iter_MC, corr_ASE, cur_case_name,
                    path_input, semipath_output, wZero_ref, wASE_ref, wVS_ref, wTINF_ref, wHist_ref, wHistAgg_ref,
                    iter_MC_ref, corr_ASE_ref, cur_case_name_ref)

'''--------------------------------------------------
                    Call to the main
--------------------------------------------------'''
## if __name__ is "__main__":
main()
