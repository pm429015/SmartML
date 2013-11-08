from numpy import *

'''
Created on Jun 21, 2013

@author: pm429015
'''


class Normalize:
    #===========================================================================
    # A module that normalize the data to 0~1 if necessary
    #===========================================================================
    def __init__(self, data):
        self.data = data

        #===========================================================================
    # Methos include : minMax, zscore and log
    #===========================================================================

    def transfor(self, types, array):
        if types == 'minMax':
            max = array.max(axis=0)
            min = array.min(axis=0)
            array = (array - min) / (max - min)
            return array
        elif types == 'zscore':
            std = array.std(axis=0)
            mean = array.mean(axis=0)
            array = (array - mean) / std
            return array
        elif types == 'log':
            array = log10(array)
            return array
        else:
            print '!! undefined normalize method'


    def normalizing(self, types, problemtype):
        # Check if the label feature is added the end of data
        nsamples, nfeatures = self.data.shape

        if problemtype == 'supervise':
            self.data[:, :nfeatures - 1] = self.transfor(types, self.data[:, :nfeatures - 1])
            #return perm1,perm2
        else:
            self.data[:, :nfeatures] = self.transfor(types, self.data[:, :nfeatures])
            #return perm1,perm2
        
        