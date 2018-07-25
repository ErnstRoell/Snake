import numpy as np
import configuration as cf


class Neural_Network(object):
    def __init__(self):
        #  Define Hyperparameters
        self.inputLayerSize = cf.inputLayerSize
        self.outputLayerSize = cf.outputLayerSize
        self.hiddenLayerSize = cf.hiddenLayerSize
        self.score = []
        self.W1 = []
        self.W2 = []

    def forward(self, X):
        #  Propogate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def mutateSelf(self):
        change = cf.change
        rate = cf.rate

        # Change W1 input array

        # random boolean mask for which values will be changed
        maskW1 = np.random.binomial(1, change, self.W1.shape).astype(np.bool)
        maskW2 = np.random.binomial(1, change, self.W2.shape).astype(np.bool)
        # random matrix the same shape of your data
        r1 = np.random.normal(0, rate, self.W1.shape)
        r2 = np.random.normal(0, rate, self.W2.shape)
        # use your mask to replace values in your input array
        self.W1[maskW1] = self.W1[maskW1] + r1[maskW1]
        self.W2[maskW2] = self.W2[maskW2] + r2[maskW2]

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 2/(1+np.exp(-z))-1
