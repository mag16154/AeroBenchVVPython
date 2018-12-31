import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from frechet import normTrajectory, norm
from itertools import combinations
import random


class NNConfiguration:

    def __init__(self):

        self.relativeError = []
        self.sortedError = []
        self.input = None
        self.output = None
        self.no_of_layers = 5
        self.neurons = [30, 30, 30, 30, 30]
        self.test_size = 0.33

    def setLayers(self, number_of_layers):
        self.no_of_layers = number_of_layers

    def setNeurons(self, neurons):
        self.neurons = neurons

    def trainTestNN(self, act_fn='relu', solver_fn='adam'):

        X_train, X_test, Y_train, Y_test = train_test_split(self.input, self.output, test_size=self.test_size, random_state=1)
        clf = None
        if self.no_of_layers == 2:
            clf = MLPRegressor(activation = act_fn, solver= solver_fn, alpha=1e-5, hidden_layer_sizes=(self.neurons[0], self.neurons[1]), random_state=1)
        elif self.no_of_layers == 3:
            clf = MLPRegressor(activation = act_fn, solver= solver_fn, alpha=1e-5, hidden_layer_sizes=(self.neurons[0], self.neurons[1], self.neurons[2]),
                               random_state=1)
        elif self.no_of_layers == 5:
            clf = MLPRegressor(activation = act_fn, solver= solver_fn, alpha=1e-5, hidden_layer_sizes=(self.neurons[0], self.neurons[1], self.neurons[2], self.neurons[3], self.neurons[4]),
                               random_state=1)

        clf.fit(X_train, Y_train)
        # train_predictions = clf.predict(X_train)
        test_predictions = clf.predict(X_test)
        print("Y_test length {} test_predictions length {}".format(len(Y_test), len(test_predictions)))

        mse = mean_squared_error(Y_test, test_predictions)
        print("Mean Squared error {}".format(mse))
        # mseError = 0.0
        # for idx in range(0, len(test_predictions)):
        #    mseError += mean_squared_error(test_predictions[idx], Y_test[idx])
        # mseError = mseError/len(test_predictions)
        # print("Manual MSE {}".format(mseError))
        relativeError = []
        for idx in range(0, len(test_predictions)):
            distVal = norm(test_predictions[idx] - Y_test[idx], -1)
            relativeError += [distVal/norm(Y_test[idx], -1)]
        plt.plot(relativeError)
        print("Mean relative error {}".format(sum(relativeError)/len(test_predictions)))
        self.relativeError = [relativeError]
        sortedError = []
        for idx in range(0, len(test_predictions)):
            sortedError += [norm(test_predictions[idx], -1)/norm(Y_test[idx], -1)]
        plt.plot(sortedError)
        self.sortedError = [sortedError]
        print("Mean sorted error {}".format(sum(sortedError) / len(test_predictions)))
        plt.show()
