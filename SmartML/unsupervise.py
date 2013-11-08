'''
Created on Jun 24, 2013

@author: pm429015
'''
import math

from numpy import *

from leaner import *


class Clustermanager:
    def methodschoose(self, method, datapoint, k):
        #=======================================================================
        # choose a learner 
        #=======================================================================
        if method == 'Kmeans':
            kmeans = Kmeans()
            model = kmeans.train(datapoint, k)
            return model


class Kmeans(Leaner):
    def distance(self, x, y, p):
        #=======================================================================
        # p as a variable to choose what kind of distance 
        # 0= Manhattan  1 = Euclidean 2 = Minkowski  N= Chebyshev
        #=======================================================================
        return power(sum(power(x - y, p)), 1.0 / p)

    def updateMatric(self, point, matric):
    #=======================================================================
    # choose a way to update the centers
    # either mean of datapoint or median of datapoint
    #=======================================================================
        if matric == 'mean':
            return mean(point, axis=0)
        elif matric == 'median':
            return median(point, axis=0)

    def train(self, datapoint, k):
        self.Nsamples, self.Nfeature = datapoint.shape

        minima = datapoint.min(axis=0)
        maxima = datapoint.max(axis=0)

        #=======================================================================
        # Pick the centres locations randomly
        #=======================================================================
        centres = random.rand(k, self.Nfeature) * (maxima - minima) + minima

        assignmentChange = True
        assignment = zeros((self.Nsamples, 1))

        #=======================================================================
        # Comparing centers with all data point, then asign the data point to the centers with min distance
        # update the centers with medium asigned datapoint
        # until assign labels are unchangeable  
        #=======================================================================
        while assignmentChange:
            assignmentChange = False

            for i in range(self.Nsamples): #data loop
                minDist = inf
                minIndex = -1
                for j in range(k): # center loop
                    currentDist = self.distance(datapoint[i], centres[j], p=2)
                    if currentDist < minDist:
                        minDist = currentDist
                        minIndex = j
                if assignment[i] != minIndex:
                    # if assign lable is change recalculate the center again
                    assignment[i] = minIndex
                    assignmentChange = True

            for label in range(k):
                cluster = where(assignment == label)[0]
                centres[label] = self.updateMatric(datapoint[cluster], matric='median')

        for s in range(len(centres)):
            if math.isnan(centres[s][0]):
                centres = self.train(datapoint, k)[0]
                print "fale try again"
            #         print centres
        return centres, assignment
         

        
        
        