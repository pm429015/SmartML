'''
Created on Jun 22, 2013

@author: pm429015
'''
from numpy import *
from numpy.linalg import *
from numpy import linalg as lin
from scipy import linalg as la


class DimReduceManager:
    def __init__(self, data, problemtype):
        self.__problemtype = problemtype
        self.__data = data

    def methods(self, method, dim, array, kernel, param):
        #=======================================================================
        # choose a method that you think the best for you
        # array = data
        #=======================================================================
        if method == 'PCA':
            Pca = PCA(array)
            datamap = Pca.map(dim)
            return datamap
        elif method == 'SVD':
            Svd = SVD(array)
            datamap = Svd.map(dim)
            return datamap
        elif method == 'LDA':
            if self.__problemtype == 'supervise':
                Lda = LDA(array)
                datamap = Lda.map(dim, self.__data[:, -1])
                return datamap
            else:
                print 'LDA currently only support supervise data';
                return None;
        elif method == 'FactorAnalysis':
            Factor = FactorAnalysis(array);
            datamap = Factor.map(dim);
            return datamap;

        elif method == 'KPCA':
            Kpca = KPCA(array)
            datamap = Kpca.map(dim, kernel, param)
            return datamap


    def methodchoose(self, method, dim, kernel, param):

        if self.__problemtype == 'supervise':
            # the data is supervise, our output data should also include original label
            datareduce = zeros((len(self.__data), dim + 1))
            datareduce[:, -1] = self.__data[:, -1]

            # start data reducing feature in the data
            datareduce[:, :dim] = self.methods(method, dim, self.__data[:, :-1], kernel, param)
        else:
            datareduce = zeros((len(self.__data), dim));
            datareduce = self.methods(method, dim, self.__data, kernel, param);

        return datareduce;


class DimReduce:
    #===========================================================================
    # this is a templet for dim reduce methods
    #===========================================================================

    def __init__(self, data):
        self._data = data
        #print self.__data

    def map(self,param):
        raise


class PCA(DimReduce):
    def map(self, N):
        means = mean(self._data, axis=0)
        data = self._data - means

        ## Covariance matrix
        covar = cov(data, rowvar=0)

        ## eigvalues, vectors
        eigvalues, eigvectors = eig(covar)

        ## sort it from big to small
        index = argsort(eigvalues)
        index = index[:-(N + 1):-1]
        eigvector_sorted = eigvectors[:, index]

        ## summary it transfor into low Dimensions
        newdata = dot(data, eigvector_sorted)

        return newdata


class SVD(DimReduce):
    def map(self, N):

        # calculate SVD
        U, s, V = linalg.svd(self._data)
        #Sig = mat(eye(N) * s[:N])

        # tak out columns you don't need
        return U[:, :N]



class LDA(DimReduce):
    def map(self, N, labels):
        means = mean(self._data, axis=0)
        data = self._data - means
        Nsamples, Nfeature = data.shape

        Sw = zeros((Nfeature, Nfeature))
        Sb = zeros((Nfeature, Nfeature))

        Cov = cov(transpose(data))

        classes = unique(labels)
        for i in range(len(classes)):
            # Find relevant datapoints
            indices = squeeze(where(labels == classes[i]))
            d = squeeze(data[indices, :])
            classcov = cov(transpose(d))
            Sw += float(shape(indices)[0]) / classcov * classcov

        Sb = Cov - Sw

        #=======================================================================
        # Now Finally solve for W
        # Compute eigenvalues, eigenvectors and sort into order
        #=======================================================================
        eigenvalues, eigenvectors = la.eig(Sw, Sb)
        indices = argsort(eigenvalues)
        indices = indices[::-1]
        eigenvectors = eigenvectors[:, indices]
        eigenvalues = eigenvalues[indices]
        w = eigenvectors[:, :N]

        #=======================================================================
        # transfor original data into new reduce dim data
        #=======================================================================
        newdata = dot(data, w)

        return newdata


class FactorAnalysis(DimReduce):
    def map(self, N):
        Nsamples, Nfeature = self._data.shape
        means = mean(self._data, axis=0)
        data = self._data - means
        C = cov(transpose(data))
        Cd = C.diagonal()
        Psi = Cd
        scaling = linalg.det(C) ** (1. / Nfeature)

        W = random.normal(0, sqrt(scaling / N), (Nfeature, N))

        nits = 1000
        oldL = -inf

        for i in range(nits):

        #=======================================================================
            # EM Algorithm : E-step
            #=======================================================================
            A = dot(W, transpose(W)) + diag(Psi)
            logA = log(abs(linalg.det(A)))
            A = linalg.inv(A)

            WA = dot(transpose(W), A)
            WAC = dot(WA, C)
            Exx = eye(N) - dot(WA, W) + dot(WAC, transpose(WA))

            #===================================================================
            # EM Algorithm : M-step
            #===================================================================
            W = dot(transpose(WAC), linalg.inv(Exx))
            Psi = Cd - (dot(W, WAC)).diagonal()

            tAC = (A * transpose(C)).sum()
            L = -Nfeature / 2 * log(2. * pi) - 0.5 * logA - 0.5 * tAC

            if (L - oldL) < (1e-4):
                print "Stop", i
                break

            oldL = L
            A = linalg.inv(dot(W, transpose(W)) + diag(Psi))
            Ex = dot(transpose(A), W)

            newdata = dot(data, Ex)

            return newdata


class KPCA(DimReduce):
    def kernelmatrix(self, kernel, param):

        if kernel == 'linear':
            return dot(self._data, transpose(self._data))
        elif kernel == 'gaussian':
            K = zeros((shape(self._data)[0], shape(self._data)[0]))
            for i in range(shape(self._data)[0]):
                for j in range(i + 1, shape(self._data)[0]):
                    K[i, j] = sum((self._data[i, :] - self._data[j, :]) ** 2)
                    K[j, i] = K[i, j]
            return exp(-K ** 2 / (2 * param[0] ** 2))
        elif kernel == 'polynomial':
            return (dot(self._data, transpose(self._data)) + param[0]) ** param[1]

    def map(self, dim, kernel, param):

        Nsamples, Nfeature = self._data.shape
        #print kernel
        #=======================================================================
        # map data into kernel matrix
        #=======================================================================
        K = self.kernelmatrix(kernel, param)

        #=======================================================================
        # Compute the transformed data and Normail them
        #=======================================================================
        D = sum(K, axis=0) / Nsamples
        E = sum(D) / Nsamples
        J = ones((Nsamples, 1)) * D
        K = K - J - transpose(J) + E * ones((Nsamples, Nsamples))

        #=======================================================================
        # start feature reduce with eigvalues, eigvectors 
        #=======================================================================
        eigvalues, eigvectors = linalg.eig(K)
        indices = argsort(eigvalues)
        indices = indices[::-1]
        eigvectors = eigvectors[:, indices[:dim]]
        eigvalues = eigvalues[indices[:dim]]

        #=======================================================================
        # normalise the eigvectors by sqr root of eigvalues
        #=======================================================================
        sqrtE = zeros((len(eigvalues), len(eigvalues)))
        for i in range(len(eigvalues)):
            sqrtE[i, i] = sqrt(eigvalues[i])

        #=======================================================================
        # Generate KPCA data
        #=======================================================================
        #newData = transpose(dot(sqrtE, transpose(eigvectors)))
        newData = dot(K, eigvectors)

        return newData

