# Created on Jun 21, 2013
# @author: pm429015

from numpy import *

from smartML.main import SmartML



#First, Declare my smart ML
my = SmartML()

# Load data
my.loadfile('./testSet.txt', ',', 'supervise', normalize=None)

# plot method if you want to visualization
my.plot(1, 0)

# dim reduce methods call
my.dimReduce('FactorAnalysis')

# call ML methods
my.learner(method='libsvm')
#my.learner('libsvm',param = '-c 1')

# test your model
my.tester(array([[2, 9], [-1, 8], [2, 7.6], [0.4, 7], [-1, 9], [6.3, 5.2]]))

#HMM example
prob = {}
prob['transition'] = array([[.6, .2, .1, .2], [.6, .05, .1, .25], [.15, .05, .6, .2], [.2, .1, .3, .4]])
prob['emission'] = array([[.2, .1, .5, .2], [.1, .4, .3, .2], [.2, .3, .5, 0], [.2, .2, .1, .5]])
prob['state'] = ['watching TV', 'Pub Night', 'Party Night', 'Study']
prob['observations'] = ['tired', 'hungover', 'scared', 'fine']

my.learner('HMM', param=prob)

obs = array([0, 3, 2, 3, 0, 1, 1])

my.tester(obs)

