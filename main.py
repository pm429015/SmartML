'''
Created on Jun 21, 2013

@author: pm429015
'''
from numpy import *
from normalize import Normalize
from plot import Plot
from DimReduce import DimReduceManager
from unsupervise import Clustermanager
from regression import Regressionmanager
from supervise import Classifymanager
from Others import Othersmanager


class SmartML(object):
    def __init__(self):
        print "smartML robot create"
        self.__type = 'others'
        self.data = None

    #===========================================================================
    # Give me the data first so that I can understand your problem
    # also define what kind of problem you are looking for 
    # problem include : supervise, unsupervise, regression, recommend, others
    #===========================================================================

    def loadfile(self, location, delimiter, problemType, normalize='minMax'):
        self.data = genfromtxt(location, delimiter=delimiter)
        self.__type = problemType
        print "data loading"

        if normalize is not None:
            self.normalize(normalize)

    #===========================================================================
    # data normalize using minMax, zscore and log
    #===========================================================================
    def normalize(self, types):
        print "Data Normalize with ", types
        normalization = Normalize(self.data)
        normalization.normalizing(types, self.__type)

    #===========================================================================
    # draw the graph given x and y
    #===========================================================================
    def plot(self, x, y):
        # assign two columns (x,y) for plot
        nofrow, noffeature = self.data.shape
        graph = Plot()
        if self.__type == 'supervise' and x < (noffeature - 1) and y < (noffeature - 1):
            graph.plotting(self.data[:, x], self.data[:, y], self.data[:, -1])
        else:
            print 'Invalid feature number'

    #===========================================================================
    # Using dimensionality Reducation methods to reduce noise
    # Methods include : PCA, SVD, LDA, FactorAnalysis
    # with Kernel methods: KPCA includes linear, gaussian and polynomial 
    #===========================================================================
    def dimReduce(self, method, dim = 2 , kernel='gaussian', param=array([3, 2])):
        DRmanager = DimReduceManager(self.data, self.__type)
        self.data = DRmanager.methodchoose(method, dim, kernel, param)

    #===========================================================================
    # Learner inclue (if you don't specific, it will run all and choose the best result for you ) 
    # unsupervise: kmeans
    # regression: Linear, local weighted linear regression (lwlr) and regression tree (regtree)
    # supervise: KNN, naive bayes (naivebay), logistic regression (logistic), ID3, adaboost and SVM (libsvm)
    # others HMM
    # k: for KNN choose top k
    # param : for SVM, check svmutil.py for detail
    # alpha, lam : for regression
    #===========================================================================
    def learner(self, method = None, k=2, thresold=0.001, r=0.5, alpha=0.1, lam=10, param='-t 2'):
        self.method = method

        if self.__type == 'unsupervise':
            clustermanager = Clustermanager()
            self.model = clustermanager.methodschoose(method, self.data, k)

        elif self.__type == 'regression':
            regressionmanager = Regressionmanager()
            self.model = regressionmanager.methodschoose(method, self.data, self.data, thresold, r, alpha, lam)

        elif self.__type == 'supervise':
            classifymanager = Classifymanager()
            self.model = classifymanager.methodschoose(method, self.data, k, param)

        elif self.__type == 'others':
            others = Othersmanager()
            self.model = others.methodschoose(method, param)


    def tester(self, testpoint):

        if self.__type == 'regression':
            regressionmanager = Regressionmanager()
            result = regressionmanager.testing(self.method, self.model, testpoint)

        elif self.__type == 'supervise':
            classifymanager = Classifymanager()
            result = classifymanager.testing(self.method, self.model, testpoint)

        elif self.__type == 'others':
            others = Othersmanager()
            result = others.testing(self.method, self.model, testpoint)

        print result
        return result


        #if __name__ == "__main__":
        #my = Main('./iris_proc.data', ',', 'supervise')
        #my.loadfile()