'''
Created on Jun 22, 2013

@author: pm429015
'''

import matplotlib.pyplot as plt
from numpy import *


class Plot:
    def __init__(self):
        self.colors = ['green', 'red', 'black', 'blue', 'magenta', 'cyan', 'yellow'];
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    #label = 0 means no label
    def plotting(self, x, y, label=None):

        if label is None:
            self.ax.scatter(x, y, color='red', linewidth=2)
        else:
            classes = list(set(label))
            Nclasses = len(classes)
            for i in range(Nclasses):
                classgroup = where(label == classes[i])
                self.ax.scatter(x[classgroup], y[classgroup], color=self.colors[i], label="class: " + str(classes[i]),
                                linewidth=2)
            self.ax.legend(loc='best')

        plt.show()
