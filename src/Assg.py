'''
Created on Mar 20, 2014

@author: flurin, quentin
'''
import numpy as np
import scipy.sparse as sps
import DNL
import os.path
import warnings


class Assignment:
    '''
    This class computes or loads the assigment matrices. If they have already been calculated, then they area
    simply loaded. This part can take some time if the matrices must be calculated from scratch.
    '''
    def __init__(self, estim_param, network):
        '''
        Constructor
        '''
        # create precomputed directory if it does not exist
        if not os.path.exists(estim_param.path_precomp):
            os.makedirs(estim_param.path_precomp)

        '''-----------------------------------------------------------------------------------------------------------------
                            Loads the link flow assignment matrix if is exists, otherwise calculates it
        -----------------------------------------------------------------------------------------------------------------'''
        if os.path.isfile(estim_param.path_precomputed_asg + "_linkFlowAssgn_data.csv") and estim_param.forceRecompAssgnment is False:
            print "Loading pre-computed flow assignment matrix",
            data = np.loadtxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_data.csv", delimiter=",")
            indices = np.loadtxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_indices.csv", dtype='int_', delimiter=",")
            indptr = np.loadtxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_indptr.csv", dtype='int_', delimiter=",")
            self.linkFlowAssgn = sps.csr_matrix((data, indices, indptr)).todense()
        else:
            print "Computing flow assignment matrix",
            self.linkFlowAssgn = np.asmatrix(DNL.compute_assg_mat(network, estim_param))
            linkFlowAssgn_csr = sps.csr_matrix(self.linkFlowAssgn)
            np.savetxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_data.csv", linkFlowAssgn_csr.data, delimiter=",")
            np.savetxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_indices.csv", linkFlowAssgn_csr.indices, fmt='%i', delimiter=",")
            np.savetxt(estim_param.path_precomputed_asg + "_linkFlowAssgn_indptr.csv", linkFlowAssgn_csr.indptr, fmt='%i', delimiter=",")
        
        self.linkFlowAssgn_prime = DNL.build_flow_assg_mat(self.linkFlowAssgn, network, estim_param, network.edges_ASE_dict)
        self.subRouteAssgn_prime = DNL.build_ODflow_assg_mat(self.linkFlowAssgn, network, estim_param, network.subroutes_VS_dict)
        self.accAssgn_prime = DNL.build_flow_assg_mat(self.linkFlowAssgn, network, estim_param, network.edges_TINF_dict)
        estim_param.print_incr_runtime()

        '''-----------------------------------------------------------------------------------------------------------------
                         Loads the occupation flow assignment matrix if is exists, otherwise calculates it
        -----------------------------------------------------------------------------------------------------------------'''
        if os.path.isfile(estim_param.path_precomputed_asg + "_acc_data.csv") and estim_param.forceRecompAssgnment is False:
            print "Loading pre-computed accumulation assignment matrix",
            data = np.loadtxt(estim_param.path_precomputed_asg + "_acc_data.csv", delimiter=",")
            indices = np.loadtxt(estim_param.path_precomputed_asg + "_acc_indices.csv", dtype='int_', delimiter=",")
            indptr = np.loadtxt(estim_param.path_precomputed_asg + "_acc_indptr.csv", dtype='int_', delimiter=",")
            self.acc = sps.csr_matrix((data, indices, indptr)).todense()
            #print "Occupation assignment matrix accumulation loaded."
        else:
            print "Computing accumulation assignment matrix",
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Prevent some annoying numerical warnings from appearing
                self.acc = np.asmatrix(DNL.compute_assg_mat_accumulation(network, estim_param))
            acc_csr = sps.csr_matrix(self.acc)
            np.savetxt(estim_param.path_precomputed_asg + "_acc_data.csv", acc_csr.data, delimiter=",")
            np.savetxt(estim_param.path_precomputed_asg + "_acc_indices.csv", acc_csr.indices, fmt='%i', delimiter=",")
            np.savetxt(estim_param.path_precomputed_asg + "_acc_indptr.csv", acc_csr.indptr, fmt='%i', delimiter=",")
        estim_param.print_incr_runtime()
