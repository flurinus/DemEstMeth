'''
Created on Jul 13, 2014

@author: flurin, nicholas
'''
import pandas as pd
import numpy as np


class TrainArrival:
    '''
    Class defining the train arrival object. Contains a mapping
    between the train length and the different usage of each access ramps.
    '''
    platformTrainTypeMap = {'shortTrain': {1: ['platform1AB', 'platform1C', 'platform1D'],
                                           3: ['shortTrainsA', 'shortTrainsB', 'shortTrainsC', 'shortTrainsD'],
                                           4: ['shortTrainsA', 'shortTrainsB', 'shortTrainsC', 'shortTrainsD'],
                                           5: ['shortTrainsA', 'shortTrainsB', 'shortTrainsC', 'shortTrainsD'],
                                           6: ['shortTrainsA', 'shortTrainsB', 'shortTrainsC', 'shortTrainsD'],
                                           7: ['shortTrainsARed', 'shortTrainsBRed', 'shortTrainsCRed', 'shortTrainsDRed'],
                                           8: ['shortTrainsARed', 'shortTrainsBRed', 'shortTrainsCRed', 'shortTrainsDRed'],
                                           9: ['platform9A', 'platform9B'],
                                           70: ['noRatio'], },
                            'longTrain': {1: ['platform1AB', 'platform1C', 'platform1D'],
                                          3: ['longTrainsA', 'longTrainsB', 'longTrainsC', 'longTrainsD'],
                                          4: ['longTrainsA', 'longTrainsB', 'longTrainsC', 'longTrainsD'],
                                          5: ['longTrainsA', 'longTrainsB', 'longTrainsC', 'longTrainsD'],
                                          6: ['longTrainsA', 'longTrainsB', 'longTrainsC', 'longTrainsD'],
                                          7: ['longTrainsARed', 'longTrainsBRed', 'longTrainsCRed', 'longTrainsDRed'],
                                          8: ['longTrainsARed', 'longTrainsBRed', 'longTrainsCRed', 'longTrainsDRed'],
                                          9: ['platform9A', 'platform9B'],
                                          70: ['noRatio'], }}
    
    sectorNames = {1: ('AB', 'C', 'D'),
                   3: ('A', 'B', 'C', 'D'),
                   4: ('A', 'B', 'C', 'D'),
                   5: ('A', 'B', 'C', 'D'),
                   6: ('A', 'B', 'C', 'D'),
                   7: ('A', 'B', 'C', 'D'),
                   8: ('A', 'B', 'C', 'D'),
                   9: ('C', 'D'),
                   70: tuple('E', ), }
    
    def __init__(self, trainNr, arrTime, track, disembarkations, Nc, TINFParamDist, alphaModel, alphaModelRed):
        '''
        Constructor. Creates an instance of a train arrival event. For the arguments passed to the constructor
        a train arrival on the corresponding platform will be created.
        '''
        self.trainNr = trainNr
        self.arrTime = arrTime
        self.track = track
        self.disemb = disembarkations
        self.Nc = Nc
        
        self.TINFParamDist = TINFParamDist
        self.alphaModel = alphaModel
        self.alphaModelRed = alphaModelRed
                
        if self.Nc <= 5:
            self.trainType = 'shortTrain'
        else:
            self.trainType = 'longTrain'
        
        if self.track in {1, 70}:
            #for platforms 1 and 70, dead time of zero
            self.dead_time = 0
        else:
            # for 'normal' platforms, dead time as specified by standard model
            self.dead_time = self.sample_dead_time()
        
        self.alpha = self.sample_cap_flow_rate(self.disemb, self.track)
            
        self.ratios = self.sample_flow_distribution(TINFParamDist.ix[self.platformTrainTypeMap[self.trainType][self.track], :], self.sectorNames[self.track])
        
    def sample_cap_flow_rate(self, Q, platform):
        '''
        Takes the data from distribution specs from normalFits and returns one alpha.
        Samples from the zero-mean distribution.
        '''
        if platform == 70:
            return self.TINFParamDist.loc['alpha70', 'mu']
        elif platform == 1:
            return self.TINFParamDist.loc['alpha1', 'mu']
        elif platform == 9:
            return self.TINFParamDist.loc['alpha9', 'mu']
        elif platform == 7 or platform == 8:
            if Q > self.alphaModelRed.loc['Qc']:
                return self.alphaModelRed.loc['a'] * self.alphaModelRed.loc['Qc'] + self.alphaModelRed.loc['b'] + self.TINFParamDist.loc['alphaFit', 'sigma'] * np.random.randn()
            elif 0 < Q < self.alphaModelRed.loc['Qc']:
                return self.alphaModelRed.loc['a'] * Q + self.alphaModelRed.loc['b'] + self.TINFParamDist.loc['alphaFit', 'sigma'] * np.random.randn()
        else:
            if Q > self.alphaModel.loc['Qc']:
                return self.alphaModel.loc['a'] * self.alphaModel.loc['Qc'] + self.alphaModel.loc['b'] + self.TINFParamDist.loc['alphaFit', 'sigma'] * np.random.randn()
            elif 0 < Q < self.alphaModel.loc['Qc']:
                return self.alphaModel.loc['a'] * Q + self.alphaModel.loc['b'] + self.TINFParamDist.loc['alphaFit', 'sigma'] * np.random.randn()

    def sample_flow_distribution(self, normalSpecs, sectorNames):
        '''
        Normalizes the distribution between the access ramps.
        The sum must be one to not "loose" any pedestrians.
        '''
        numEntries = normalSpecs.shape[0]
        rawRatios = normalSpecs.ix[:, 'sigma'] * (pd.Series(np.array(np.random.randn(numEntries)), index=normalSpecs.index.values)) + normalSpecs.ix[:, 'mu']
        rawRatios[rawRatios < 0] = 0
        ratios = rawRatios / np.sum(rawRatios)
        ratios.index = sectorNames
        return ratios

    def sample_dead_time(self):
        '''
        Returns a scalar from the normal distribution with parameters specified in normalFits DataFrame
        '''
        dead_time = -1
        while dead_time < 0:
            dead_time = self.TINFParamDist.loc['s', 'sigma'] * np.random.randn() + self.TINFParamDist.loc['s', 'mu']
        return dead_time
