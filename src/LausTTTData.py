'''
Created on 26 Mar 2014

@author: nicholas, flurin

'''
import pandas as pd
import numpy as np
import datetime
from scipy import integrate
import warnings
from Train import TrainArrival


## class containing the data and the final time space expanded vector
class TINFLinkCount(object):
    '''
    Class containing the required methods and objects to estimate train induced flows.
    Based on the train timetables and train occupation surveys the flows induced by
    each train is estimated.
    '''
    def __init__(self, estim_param, network):
        '''
        Constructor. This method creates an instance of the train induced flows based on
        the available data. It is stochastic process so one single draw is not acceptable.
        '''
        np.random.seed()  # Required for parallel computing
        
        # load timetable and train frequency data
        self.OTT = pd.read_csv(estim_param.path_OTT)
        self.FQ = pd.read_csv(estim_param.path_FQ)
        
        # load parameter specification of TINF model
        self.TINFParamDist = pd.read_csv(estim_param.path_TINFParamDist)
        self.alphaModel = pd.read_csv(estim_param.path_alphaModel).ix[:, 0]
        self.alphaModelRed = pd.read_csv(estim_param.path_alphaModelRed).ix[:, 0]
        
        # generate list of trains
        trainList = self.TrainCollection(estim_param, self.OTT, self.FQ, self.TINFParamDist, self.alphaModel, self.alphaModelRed)
                
        self.TINF = self.ParseAllTrains(trainList, estim_param, network)
        
    def TrainCollection(self, estim_param, OTT, FQ, TINFParamDist, alphaModel, alphaModelRed):
        '''
        From the csv files containing the time tables, passenger surveys and train specific parameters
        the collection of trains is generated. Each train has a list of parameters.
        '''
        trainCollection = {}
        for i in FQ.index.values:
            if isinstance(FQ.ix[i, 't_arr_sched'], str):
                if np.isnan(FQ.ix[i, 'arr_HOP']):
                    numberDisemPass = -1
                    while(numberDisemPass <= 0):
                        numberDisemPass = (FQ.ix[i, 'arr_FRASY'] + np.random.randn() * estim_param.FQ_std * FQ.ix[i, 'arr_FRASY'])
                else:
                    numberDisemPass = -1
                    while(numberDisemPass <= 0):
                        numberDisemPass = FQ.ix[i, 'arr_HOP'] + np.random.randn() * estim_param.FQ_std * FQ.ix[i, 'arr_HOP']
                
                is_trainNr = OTT['trainNr'] == FQ.ix[i, 'trainNr']
                is_year = OTT['year'] == estim_param.start_date_t.year
                is_month = OTT['month'] == estim_param.start_date_t.month
                is_day = OTT['day'] == estim_param.start_date_t.day

                cur_train = OTT[is_trainNr & is_year & is_month & is_day]
                
                arr_time = datetime.time(int(cur_train['hr_arr']), int(cur_train['min_arr']), int(cur_train['sec_arr']))
                
                trainCollection[i] = TrainArrival(FQ.ix[i, 'trainNr'], arr_time, cur_train['track'].values[0], numberDisemPass, FQ.ix[i, 'Nc'], TINFParamDist, alphaModel, alphaModelRed)

        return trainCollection
            
    def ParseAllTrains(self, trainCollection, estim_param, network):
        '''
        Goes through the train collection and fills the appropriate
        links in the network with the pedestrian flows.
        '''
        tint_dict = estim_param.tint_dict
        edges_TINF_dict = network.edges_TINF_dict
        edges_TINF_origins = network.edges_TINF_origins
        
        tinf = np.zeros(len(edges_TINF_dict) * len(tint_dict))
        
        for train in trainCollection:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # prevent some annoying numerical warnings from appearing
                tinfEstimate = self.unloadingLinkCountEstimate(tint_dict, trainCollection[train])
            
            for sec in tinfEstimate.columns.values.tolist():
                pos = [key for key in edges_TINF_dict.keys() if (str(trainCollection[train].track) in edges_TINF_origins[key] and sec in edges_TINF_origins[key])]
                tinf[range(pos[0], len(tinf), len(edges_TINF_dict))] = tinf[range(pos[0], len(tinf), len(edges_TINF_dict))] + tinfEstimate.ix[:, sec]  # Requires numpy v1.8
        return tinf
            
    def Heaviside(self, x):
        '''
        Implementation of the heaviside function
        '''
        return 0.5 * (np.sign(x) + 1)

    def PWL(self, timeSeries, tm, s, Q, alpha):
        '''
        Implementation of the PieceWise Linear model for disembarking flows
        '''
        return self.Heaviside(timeSeries - tm - s) * alpha * self.Heaviside(- (timeSeries - tm - s - (Q / alpha)))

    def unloadingLinkCountEstimate(self, tint_dict, trainObject):
        '''
        Aggregates the flows into the time intervals specified in the mian class
        '''
        tint_dictComplete = dict(tint_dict)
        
        tint_dictComplete.update({len(tint_dict): (datetime.datetime.strptime(tint_dict[len(tint_dict) - 1], "%d-%b-%Y %H:%M:%S") +
                                                   (datetime.datetime.strptime(tint_dict[1], "%d-%b-%Y %H:%M:%S") -
                                                    datetime.datetime.strptime(tint_dict[0], "%d-%b-%Y %H:%M:%S"))).strftime("%d-%b-%Y %H:%M:%S")})
        
        indexNames = [x for i, x in enumerate(tint_dictComplete.values()) if i != len(tint_dictComplete) - 1]

        linkCounts = pd.DataFrame(index=indexNames, columns=trainObject.ratios.index.values)
        
        tmDateTime = datetime.datetime.strptime(tint_dictComplete[0][0:12] + str(trainObject.arrTime), "%d-%b-%Y %H:%M:%S")
        
        for i in range(len(tint_dictComplete) - 1):
            tauTotalPed = integrate.quad(self.PWL,
                                         (datetime.datetime.strptime(tint_dictComplete[i], "%d-%b-%Y %H:%M:%S") - datetime.datetime.strptime(tint_dictComplete[0], "%d-%b-%Y %H:%M:%S")).total_seconds(),
                                         (datetime.datetime.strptime(tint_dictComplete[i + 1], "%d-%b-%Y %H:%M:%S") - datetime.datetime.strptime(tint_dictComplete[0], "%d-%b-%Y %H:%M:%S")).total_seconds(),
                                         args=((tmDateTime - datetime.datetime.strptime(tint_dictComplete[0], "%d-%b-%Y %H:%M:%S")).total_seconds(), trainObject.dead_time, trainObject.disemb, trainObject.alpha))
            linkCounts.ix[i, :] = tauTotalPed[0] * trainObject.ratios
        return linkCounts
