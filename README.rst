.. -*- mode: rst -*-

About
=====

Welcome to my Machine Learning playground. SmartML is a collection of experiment machine learning algorithms that I implemented from scratch in python. The goal of this project is to get hands dirty and experience the real power of Machine Learning. 

My ultimate goal is to build a ML toolbox that speed up the time on finding the best approach
for your dataset. This is the reason why I call it ‘smart’ ML.

Currently, this project is very far from  my ultimate goal, but it is good for anyone who want to play around the well-known ML algorithms. You are free to change or use it, but beware some unexpected bugs since I am still working on it. 

My blog also cover most of ML algorithms that I implemented in SmartML with examples, so feel free stop by. Thanks.

My Blog : `Make it easy`_

.. _`Make it easy`: http://pm429015.wordpress.com/



Project Setup:
==============

Requirement:

- Python 2.75
- Numpy
- Matplotlib
- Libsvm (Please Go to : `LIBSVM`_ to Download the library, then open a terminal direct to .libsvm.3.XX/python/ and type “make”.)

.. _`LIBSVM`: https://github.com/cjlin1/libsvm


SmartML Algorithm include :
============

Dimension reduction:

- PCA
- Kernel PCA
- SVD
- LDA
- Factor Analysis

Normalization:

- min MAX
- Z Score
- log

Supervise Algorithms (Classifers):

- KNN
- Logistic Regression
- ID3 Tree
- Naive Bayes 
- SVM
- Adaboost


Unsupervise Algorithm :

- KMean

Regression :

- RidgeRegression
- Local weighted linear regression (lwlr) 
- Regression tree (regtree)

Others: 

- Hidden Markov Model


Usage
============


from smartML.main import SmartML

# General Usage

1. First, Declare my smart ML ::

	myML = SmartML()

2. Load data using trainSet.txt for example::

	myML.loadfile('./trainSet.txt', ',', 'supervise', normalize=‘minMax’)

3. plot data (Optional)::

	myML.plot(1, 0)

4. dim reduce methods call (Optional)::

	myML.dimReduce('FactorAnalysis')

5. call ML methods::

    # Methods Include: 
    # unsupervise: kmeans
    # regression: Linear, local weighted linear regression (lwlr) and regression tree (regtree)
    # supervise: KNN, naive bayes (naivebay), logistic regression (logistic), ID3, adaboost and SVM (libsvm)

	myML.learner(method=‘ID3’)

6. test your model ::
	myML.tester(test.dataset)


7. return a result label array for testing dataset

# HMM

1. construct a probability table::

	prob = {}

	prob['transition'] = array([[.6, .2, .1, .2], [.6, .05, .1, .25], [.15, .05, .6, .2], [.2, .1, .3, .4]])
	
	prob['emission'] = array([[.2, .1, .5, .2], [.1, .4, .3, .2], [.2, .3, .5, 0], [.2, .2, .1, .5]])
	
	prob['state'] = ['watching TV', 'Pub Night', 'Party Night', 'Study']
	
	prob['observations'] = ['tired', 'hungover', 'scared', 'fine']

2. call smartML::

	myML = SmartML()::

3. call leaner method with table::

	my.learner('HMM', param=prob)

4. create a observation::

	obs = array([0, 3, 2, 3, 0, 1, 1])

5. probability return::

	my.tester(obs)

`My Blog for HMM silly example`_

.. _`My Blog for HMM silly example`: http://pm429015.wordpress.com/2013/05/21/hmm/


Have Fun !~


