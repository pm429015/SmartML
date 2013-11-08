'''
Created on Jun 27, 2013

@author: pm429015
'''
from numpy import *
from leaner import *


class Regressionmanager:
    
    #=======================================================================
    # add ones into datapint for regression calculation 
    #=======================================================================
    def addone(self, data):
        datamat = mat(data)
        m, n = datamat.shape
        
        if  m == 1:  # which means one one feature
            Nsample = shape(datamat)[1]
            one = mat(ones(Nsample))
            X = hstack((one.T, datamat.T))
        else:
            Nsample = shape(datamat)[0]
            one = mat(ones(Nsample))
            X = mat(hstack((one.T, datamat)))
        return X

    def methodschoose(self, method, datapoint, pred , thresold, r, alpha, lam):
        
        #=======================================================================
        # choose a learner 
        #=======================================================================
        if method == 'RidgeRegression':
            datapoint = self.addone(datapoint)
            ridge = RidgeRegression()
            model = ridge.train(datapoint, pred, thresold, alpha, lam)
             
        elif method == 'lwlr':
            datapoint = self.addone(datapoint)
            lwlr = Lwlr()
            model = lwlr.train(datapoint, pred, r)
            
        elif method == 'regtree':
            datapoint = array(self.addone(datapoint))
            data = vstack((datapoint.T, pred)).T
            regtree = Regtree()
            model = regtree.train(data)
        return model
    
    #===========================================================================
    # we use testing function to test the given points using exist trained model
    #===========================================================================
    def testing(self, method, model, testpoint):
        testpoint = self.addone(testpoint)

        if method == 'RidgeRegression':
            ridge = RidgeRegression()
            result = ridge.test(model, testpoint)
        elif method == 'lwlr':
            lwlr = Lwlr()
            result = lwlr.test(model, testpoint)
        elif method == 'regtree':
            regtree = Regtree()
            result = regtree.test(model, testpoint)
        return result
    
    
class RidgeRegression(Leaner):
    
    #=======================================================================
    # graddescent optimatizer invlove
    #=======================================================================
    def GradDescent(self, x, y, alpha, lam, loop=50):
        theta = ones((self.Nfeature , 1))
        for i in xrange(loop):
            theta = theta - (theta * alpha * lam) / self.Nsamples - alpha / self.Nsamples * x.T * (x * theta - y.T)
    
        return theta
    
    def test(self, theta, x):
        y = theta.T * x.T
        return y
    
    #===========================================================================
    # calculate the cost of the model
    # if cost is less than k, early stop
    #===========================================================================
    def train(self, x, pred, thresold, alpha, lam):

        self.Nsamples, self.Nfeature = x.shape
        y = mat(pred)
        n = 1000
        tempcost = inf
         
        for i in xrange(n):
            theta = self.GradDescent(x, y, alpha, lam, i)
            ypred = self.test(theta, x)
            cost = abs(y - ypred).sum()
            if (tempcost - cost) > thresold:
                tempcost = cost
            else:
                print 'grad descent loop stop until %d' % i
                break
            if i == n:
                print "can't coverage lamda add 5 run again"
                self.train(x, pred, thresold, alpha, lam + 5)
  
        return theta

class Lwlr(Leaner):
    #===========================================================================
    # RBF kernel uses in this case 
    #===========================================================================
    def kernel(self, point, X, r):
        m, n = shape(X)
        weights = mat(eye((m)))
        for j in range(m):
            diff = point - X[j]
            weights[j, j] = exp(diff * diff.T / (-2.0 * r ** 2))
        return weights

    def localWeight(self, point, X, pred, r):
        wei = self.kernel(point, X, r)
        W = (X.T * (wei * X)).I * (X.T * (wei * pred.T))
        return W
    
    def train(self, X, pred, r):
        matpred = mat(pred)
        model = {}
        model[0] = X
        model[1] = matpred
        model[2] = r
         
        m, n = shape(X)
        ypred = zeros(m)
         
        for i in xrange(m):
            ypred[i] = X[i] * self.localWeight(X[i], X, matpred, r)
            
        return model
     
    def test(self, model, X):
        m, n = shape(X)
        y = zeros(m)
         
        for i in xrange(m):
            y[i] = X[i] * self.localWeight(X[i], model[0], model[1], model[2])
          
        return y
        
class Regtree(Leaner): 
    #===========================================================================
    # error calcuate 
    #===========================================================================
    def totalErr(self, dataSet):
        return var(dataSet[:, -1]) * shape(dataSet)[0]
    
    def split(self, data, feature, threshold):  # target data, which feature split, threshold
        biggerleaf = data[where(data[:, feature] > threshold)[0]]
        smallerleaf = data[where(data[:, feature] <= threshold)[0]]
        return biggerleaf, smallerleaf
    
    #===========================================================================
    # find the best feature and threshold to split
    #===========================================================================
    def bestspilt (self, data, minaSample=4 , minaerror=1):

        if (sum(data[:, -1]) / len(data[:, -1])) == data[0, -1]: 
            #  if target variable is equal, nothing is needed to compute
            return None, None, None, mean(data[:-1])
        
        originalerror = self.totalErr(data)
        Nsamples , Nfeatures = data.shape
        minerror = inf 
        bestfeature = 0 
        bestthreshold = 0
        
        for i in xrange(Nfeatures - 1):  # for every features
            for j in set(data[:, i]):  # loop every scales as possible threshold
                bigD, smallD = self.split(data, i, j)
                # skip if don't have much data
                if (shape(bigD)[0] > minaSample) and (shape(smallD)[0] > minaSample): 
                    errorsum = self.totalErr(bigD) + self.totalErr(smallD)
                    if errorsum < minerror: 
                        bestfeature = i
                        bestthreshold = j
                        minerror = errorsum
        if (originalerror - minerror) < minaerror:
            return None, None, None, mean(data[:, -1])
        biger, smaller = self.split(data, bestfeature, bestthreshold)
        if (shape(biger)[0] < minaSample) or (shape(smaller)[0] < minaSample): 
            return None, None, None, mean(data[:, -1])
      
        return biger, smaller, bestfeature, bestthreshold

    
    #===========================================================================
    # loop bestspilt until points less than 5
    #===========================================================================
    def train(self, data):
        biger, smaller, bestfeature, bestthreshold = self.bestspilt(data, 5, 0.2)
        if biger == None: 
            return bestthreshold
        else:
            tree = {}
            tree['which feature split '] = bestfeature
            tree['threshold'] = bestthreshold
            tree['bigger leaf'] = self.train(biger)
            tree['smaller leaf'] = self.train(smaller)
           
            return tree

    def isTree(self, obj):
        return (type(obj).__name__ == 'dict')

    def getMean(self, tree):
        if self.isTree(tree['bigger leaf']): tree['bigger leaf'] = self.getMean(tree['bigger leaf'])
        if self.isTree(tree['smaller leaf']): tree['smaller leaf'] = self.getMean(tree['smaller leaf'])
        return (tree['smaller leaf'] + tree['bigger leaf']) / 2.0
    
    #===========================================================================
    # to avoid overfitting, prune out some split that is less important
    #===========================================================================
    def prune(self, tree, testData):
        if shape(testData)[0] == 0: return self.etMean(tree)  # if we have no test data collapse the tree
        if (self.isTree(tree['bigger leaf']) or self.isTree(tree['smaller leaf'])):  # if the branches are not trees try to prune them
            smaller, bigger = split(testData, tree['which feature split '], tree['threshold'])
        if self.isTree(tree['smaller leaf']): tree['smaller leaf'] = self.prune(tree['smaller leaf'], smaller)
        if self.isTree(tree['bigger leaf']): tree['bigger leaf'] = self.prune(tree['bigger leaf'], bigger)
        # if they are now both leafs, see if we can merge them
        if not self.isTree(tree['smaller leaf']) and not self.isTree(tree['bigger leaf']):
            smaller, bigger = split(testData, tree['which feature split '], tree['threshold'])
            errorNoMerge = sum(power(smaller[:, -1] - tree['smaller leaf'], 2)) + sum(power(bigger[:, -1] - tree['bigger leaf'], 2))
            treeMean = (tree['smaller leaf'] + tree['bigger leaf']) / 2.0
            errorMerge = sum(power(testData[:, -1] - treeMean, 2))
            if errorMerge < errorNoMerge: 
                print "merging"
                return treeMean
            else: return tree
        else: return tree

    def test(self, tree, testD):
        """ Function doc """

        answer = ones((len(testD)))
        for i in range(len(testD)):
            if type(tree) is dict:
               
                if testD[i, tree['which feature split ']] > tree['threshold']:
                    answer[i] = self.test(tree['bigger leaf'], testD[i])
                else:
                    answer[i] = self.test(tree['smaller leaf'], testD[i])
            else:
                return tree
            
        return answer
    
