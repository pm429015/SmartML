'''
Created on Jul 11, 2013

@author: pm429015
'''

from numpy import *


class Othersmanager:
    def methodschoose(self, method, param):

        if method == 'HMM':
            hmm = HMM()
            model = hmm.train(param)

        return model

    def testing(self, method, model, testpoint):
        if method == 'HMM':
            hmm = HMM()
            result = hmm.test(model, testpoint)

        return result


class HMM:
    def train(self, param):
        return param

    def test(self, model, testpoint):
        states = shape(model['emission'])[0]
        TimesOfObserb = shape(testpoint)[0]
        alpha = zeros((states, TimesOfObserb))
        answer = ()
        #given the first state
        firstState = ones((len(model['state'])))

        alpha[:, 0] = firstState / len(firstState)
        for t in range(1, TimesOfObserb):
            for s in range(states):
                alpha[s, t] = model['emission'][s, testpoint[t]] * sum(alpha[:, t - 1] * model['transition'][:, s])
            alpha[:, t] = alpha[:, t] / sum(alpha[:, t])

        for i in xrange(1, TimesOfObserb):
            answer = append(answer, model['state'][where(alpha.T[i] == max(alpha.T[i]))[0]])

        return answer
