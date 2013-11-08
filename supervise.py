'''
Created on Jul 6, 2013

@author: pm429015
'''
import operator
from collections import Counter
import math

from numpy import *

from leaner import *
from svmutil import *


class Classifymanager:
    def addone(self, data):
        datamat = mat(data)
        m, n = datamat.shape

        if m == 1:  # which means one one feature
            Nsample = shape(datamat)[1]
            one = mat(ones(Nsample))
            X = hstack((one.T, datamat.T))
        else:
            Nsample = shape(datamat)[0]
            one = mat(ones(Nsample))
            X = mat(hstack((one.T, datamat)))
        return X

    def methodschoose(self, method, datapoint, k, param):
        #=======================================================================
        # choose a learner 
        #=======================================================================
        if method == 'KNN':
            knn = KNN()
            model = knn.train(datapoint, k)

        elif method == 'logistic':
            datapoint = self.addone(datapoint)
            logistic = Logistic()
            model = logistic.train(datapoint)

        elif method == 'ID3':
            id3 = ID3()
            model = id3.train(datapoint)

        elif method == 'naivebay':
            naive = Naive()
            model = naive.train(datapoint)

        elif method == 'adaboost':
            adaboost = Adaboost()
            model = adaboost.train(datapoint)

        elif method == 'libsvm':
            svm = SVM()
            model = svm.train(datapoint, param)

        return model

        #===========================================================================

    # we use testing function to test the given points using exist trained model
    #===========================================================================
    def testing(self, method, model, testpoint):

        if method == 'KNN':
            knn = KNN()
            result = knn.test(model, testpoint)
        elif method == 'logistic':
            testpoint = self.addone(testpoint)
            logistic = Logistic()
            result = logistic.test(model, testpoint)
        elif method == 'ID3':
            id3 = ID3()
            result = id3.test(model, testpoint)
        elif method == 'naivebay':
            naive = Naive()
            result = naive.test(model, testpoint)
        elif method == 'adaboost':
            adaboost = Adaboost()
            result = adaboost.test(model, testpoint)

        elif method == 'libsvm':
            svm = SVM()
            result = svm.test(model, testpoint)
        return result


class KNN(Leaner):
    def distance(self, x, y, p):
        #=======================================================================
        # p as a variable to choose what kind of distance 
        # 0= Manhattan ; 1 = Euclidean; 2 = Minkowski ; N= Chebyshev
        #=======================================================================
        return power(sum(power(x[:, :-1] - y, p), axis=1), 1.0 / p)

    def train(self, datapoint, k):
        model = {}
        model[0] = datapoint
        model[1] = k

        return model

    def test(self, model, test):
        #=======================================================================
        # calculate the distance of every points to others, then sort the distance and choose top k points 
        # finally label was decided by top k labels
        #=======================================================================
        answer = ones((len(test)))

        for i in range(len(test)):
            # Distance with trainSet (p = which distance method)
            distance = self.distance(model[0], test[i], p=2)
            # sort small to big into a list
            sortedDistIndicies = distance.argsort()

            # create a dectionary and store vote 
            voteIlabel = model[0][sortedDistIndicies[:model[1]], -1]
            b = Counter(voteIlabel)
            answer[i] = b.most_common(1)[0][0]

        return answer


class Logistic(Leaner):
    def sigmoid(self, X):
        return 1.0 / (1 + exp(-X))


    def train(self, datapoint):
        #=======================================================================
        # batch version of gradient ascent approach to find the best theta
        #=======================================================================
        m, n = shape(datapoint)
        theta = mat(ones((n - 1))).T
        iter = 1000000
        lam = 0.0
        alpha = 0.01
        for i in xrange(iter):
            thetaOld = theta.copy()
            theta = theta - ((theta * alpha * lam) / m) - alpha / m * datapoint[:, :-1].T * (
            self.sigmoid(datapoint[:, :-1] * theta) - datapoint[:, -1])
            if sum(theta - thetaOld) > -0.001:
                print i
                return theta
            if i == iter - 1:
                print "sorry theta can't coverage"
                #===========================================================================
                # (this version isn't perform well)
                # stochastic gradient ascent approach to find the maxium weights
                # adjust alpha to avoid the oscillations and make sure new data still have some impact
                # take from Machine learning in action
                #===========================================================================
            #         m,n = shape(datapoint);
            #         weights = mat(ones((n-1)));
            #         numIter= 2000;
            #         for j in range(numIter):
            #             dataIndex = range(m);
            #             for i in range(m):
            #                 alpha = 4/(1.0+j+i)+0.0001;    #alpha decreases with iteration, does not
            #                 randIndex = int(random.uniform(0,len(dataIndex)));
            #                 h = self.sigmoid(sum(datapoint[randIndex,:-1]*weights.T))
            #                 error = datapoint[randIndex,-1] - h
            #                 weights = weights + alpha * error * datapoint[randIndex,:-1]
            #                 del(dataIndex[randIndex])
            # #             print str(j) +' loops has: '
            # #             print weights
            #         return weights.T;

    def test(self, model, test):
        answer = ones((len(test)))
        for i in range(len(test)):
            prob = self.sigmoid(sum(test[i] * model))
            if prob > 0.5:
                answer[i] = 1.0
            else:
                answer[i] = 0.0
        return answer


class ID3(Leaner):
    def entropy(self, prob):
        return -prob * log2(prob)

    def gain(self, data):
        # get number of samples and features , initiallize small entropy
        Nsamples = len(data)
        est = 0.0
        Nclasses = Counter(data)
        classeskey = Nclasses.keys()
        for i in xrange(len(classeskey)):
            est += self.entropy(Nclasses[classeskey[i]] / float(Nsamples))
        return est

    def compare_gain(self, data):
        #=======================================================================
        # deal with continuous variable or scales
        # each time loop a feature threshold with random half number of samples size
        # and find a best threshold that in the which feature, then output it
        #=======================================================================
        Nsamples, Nfeatures = data.shape
        smallgain = 1.0
        bestfeature = 0
        bestthreshold = 0.0
        bigger = array([])
        smaller = array([])
        step = Nsamples / 2
        for i in xrange(Nfeatures - 1):  # number of feature
            if Nsamples == len(set(data[:, i])):  # continus variable
                rangeMin = data[:, i].min()
                rangeMax = data[:, i].max()
                stepSize = (rangeMax - rangeMin) / step

                for j in range(step):
                    est = 0.0
                    threshold = (rangeMin + float(j) * stepSize)
                    upper = data[where(data[:, i] > threshold)[0]]
                    lower = data[where(data[:, i] <= threshold)[0]]
                    est = (len(upper) / float(Nsamples)) * self.gain(upper[:, -1]) + (
                    len(lower) / float(Nsamples)) * self.gain(lower[:, -1]);
                    if smallgain > est:
                        smallgain = est
                        bestfeature = i
                        bestthreshold = threshold
                        bigger = upper.copy()
                        smaller = lower.copy()
        return bestfeature, bestthreshold, bigger, smaller

    def train(self, datapoint):
        #=======================================================================
        # create a tree to store feature, threshold, bigger points and smaller points
        # until given points have same class
        #=======================================================================
        Nsamples, Nfeatures = datapoint.shape
        # count items and high vote
        labelcount = Counter(datapoint[:, -1])
        guess = labelcount.most_common(1)[0][0]

        # no feature or have same label
        if labelcount[guess] == Nsamples:
            return guess
        else:
            whichfeature, whichthreshold, bigger, smaller = self.compare_gain(datapoint)
            tree = {}
            tree['feature'] = whichfeature
            tree['threshold'] = whichthreshold
            tree['bigger'] = self.train(bigger)
            tree['smaller'] = self.train(smaller)

            return tree

    def classify(self, tree, testD):
        # recursive go through the answer
        if type(tree) is dict:
            if testD[tree['feature']] <= tree['threshold']:
                answer = self.classify(tree['smaller'], testD)
            else:
                answer = self.classify(tree['bigger'], testD)
            return answer
        else:
            return tree


    def test(self, tree, testD):
        #=======================================================================
        # recursive go through the answer
        # each test points go through the tree model 
        # until to the end of leaf, which is class 
        #=======================================================================
        answer = ones((len(testD)));

        for i in range(len(testD)):
            answer[i] = self.classify(tree, testD[i])

        return answer;


class Naive(Leaner):
    #===========================================================================
    # this naive classifier can handle two type of features discrete (0,1,2), continue(1.23,1.24,3.2,...):
    # when features are continue, I use normal equation to find the probability
    # when features are discrete, I follow the general way by counting the frequency 
    # Also, this classifier can handle these two type of features in a dataset
    #===========================================================================
    def normpdf(self, x, mean, var):
    #         var = float(sd)**2
        pi = 3.1415926
        denom = (2 * pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    def train(self, datapoint):

        Nsamples, Nfeatures = datapoint.shape
        model = {}
        type = ones((Nfeatures - 1)) # store type of feature , zero = continuous ; one = discrete
        # first calculate label numbers
        labelcount = Counter(datapoint[:, -1])
        labelcomm = labelcount.most_common()
        labelrank = ones((len(labelcomm)))
        disindex = {}
        for i in range(Nfeatures - 1):
            count = Counter(datapoint[:, i])
            if len(count) > Nsamples * 2 / 3:  #continuous variable
                static = ones((2, len(labelcomm)))  # a variable that store mean variable in continuous feature
                for j in range(len(labelcomm)):
                    labelrank[j] = labelcomm[j][0]
                    part = where(datapoint[:, -1] == labelcomm[j][0])
                    static[0][j] = mean(datapoint[part, i]) # calculate mean
                    static[1][j] = var(datapoint[part, i]) # calculate variable
                disindex[i] = 1
                type[i] = 0.0
            else:
                scalecount = count.most_common()
                static = ones((len(scalecount), len(labelcomm)))

                for j in range(len(labelcomm)): # label group
                    labelrank[j] = labelcomm[j][0]
                    index = {} #the index of scale
                    for h in range(len(scalecount)): # scale in feature count
                        index[scalecount[h][0]] = h
                        part = where(datapoint[:, -1] == labelcomm[j][0])
                        static[h][j] = (len(where(datapoint[part, i] == scalecount[h][0])[0]) + 1.0) / (
                        labelcomm[j][1] + 2.0)
                disindex[i] = index
                type[i] = 1.0
            model[i] = static
        model['type'] = type
        model['rank'] = labelrank
        model['index'] = disindex
        return model

    def classify(self, type, model, testpoint, index):
        #=======================================================================
        # As mentioned at beginning, 
        # 0 discrete variable will grap prob directly from model
        # 1 continue variable will sent it to gaussian prob
        # return the prob belong to lables
        #=======================================================================

        m, n = model.shape
        back = ones((n, 1))

        if type == 0.0:
            for i in range(n):
                back[i] = self.normpdf(testpoint, model[0, i], model[1, i])
        elif type == 1.0:
            for i in range(n):
                back[i] = model[index[testpoint], i]

        return log(back)


    def test(self, model, testD):
        number_of_class = model[0].shape[1]
        answer = ones((len(testD)))
        for i in range(len(testD)): # testpoints level
            result = ones((number_of_class, 1))
            for j in range(len(testD[i])): # feature level
                result += self.classify(model['type'][j], model[j], testD[i][j], model['index'][j])
            answer[i] = model['rank'][argmax(result)]

        return answer


class Adaboost(Leaner):
    def stumpClassify(self, data, dimen, threshVal, threshIneq):
        retArray = ones((shape(data)[0], 1))
        if threshIneq == 'lt':
            retArray[data[:, dimen] <= threshVal] = -1.0
        else:
            retArray[data[:, dimen] > threshVal] = -1.0
        return retArray

    def bestspilt(self, data, step, weight):
        labelmat = mat(data[:, -1]).T
        Nsamples, Nfeatures = shape(data)
        minerror = inf
        bestStump = {}
        bestClasEst = mat(zeros((Nsamples, 1)))

        for i in range(Nfeatures - 1):

            rangeMin = data[:, i].min()
            rangeMax = data[:, i].max()
            stepSize = (rangeMax - rangeMin) / step

            for j in range(step):
                for inequal in ['lt', 'gt']: # loop greater or smaller
                    threshold = (rangeMin + float(j) * stepSize) # calculate the threshold 

                    predictedVals = self.stumpClassify(data, i, threshold, inequal) # use a weak classifier

                    errArr = mat(ones((Nsamples, 1)))

                    errArr[predictedVals == labelmat] = 0  # change zero when result is right

                    weightedError = weight.T * errArr

                    if weightedError < minerror:
                        minerror = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshold
                        bestStump['ineq'] = inequal
        return bestStump, minerror, bestClasEst


    def train(self, dataSet):
        #=======================================================================
        # Adaboost uses weights for training data points 
        # In the beginning, all points have equal weights "1" 
        # Then, using a weak classifier to separate the data.
        # The new weights will be updated according to the result
        # The next classifiers will separate data according to current weight until the error is minizium
        #=======================================================================

        # if labels isn't (1,-1), I need to change it to (1,-1)
        data = dataSet.copy()
        labels = Counter(data[:, -1]).keys()
        index = {}
        index[-1.0] = labels[0]
        index[1.0] = labels[1]
        firstlabel = where(data[:, -1] == labels[0])
        secondlabel = where(data[:, -1] == labels[1])
        data[firstlabel, -1] = -1.0
        data[secondlabel, -1] = 1.0

        maxlearners = 30
        Nsamples, Nfeatures = shape(data)
        weight = ones((Nsamples, 1)) / Nsamples
        train_modle = []
        classEst = zeros((Nsamples, 1))

        # loop weak learning 
        for i in xrange(maxlearners):
            model_dec = {}
            bestStump, error, classEst = self.bestspilt(data, 10, weight)  #build Stump
            alpha = float(0.5 * log(
                (1.0 - error) / max(error, 1e-16))); #calc alpha, throw in max(error,eps) to account for error=0
            bestStump['alpha'] = alpha
            train_modle.append(bestStump)                 #store Stump Params in Array
            # the equation for adaboost
            expon = multiply(-1 * alpha * data[:, -1].reshape((Nsamples, 1)), classEst)
            weight = multiply(exp(expon), weight)
            #normailize
            weight = weight / weight.sum()

            classEst += alpha * classEst
            errors = where(sign(classEst) != data[:, -1].reshape((Nsamples, 1)), 1, 0)
            errorate = errors.sum() / float(Nsamples)
            if errorate == 0.0:
                break
        print "number of week classifiers %d" % maxlearners
        train_modle.append(index)
        return train_modle

    def test(self, model, testdata):
        #=======================================================================
        # push testing dataset to weak classifiers, then multipy the result with alpha
        # we choose sign of result to assign the labels 
        #=======================================================================
        Nsample, Nfeature = testdata.shape
        ClassEst = zeros((Nsample, 1))
        answer = ones((len(testdata)))
        for j in range(len(testdata)):
            for i in xrange(len(model) - 1):
                classEst = self.stumpClassify(mat(testdata[j]), model[i]['dim'], model[i]['thresh'],
                                              model[i]['ineq'])#call stump classify
                ClassEst += model[i]['alpha'] * classEst
            answer[j] = model[-1][sign(ClassEst)[0, 0]]
        return answer


class SVM(Leaner):
    def train(self, dataSet, param):
        prob = svm_problem(dataSet[:, -1].tolist(), dataSet[:, :-1].tolist())
        Setparam = svm_parameter(param)
        model = svm_train(prob, Setparam)

        return model

    def test(self, model, dataSet):
        p_labels, p_vals = svm_predict(dataSet.tolist(), model)

        return array(p_labels);
        
           