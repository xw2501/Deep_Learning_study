from __future__ import print_function

import numpy as np
from ecbm4040.classifiers.linear_svm import *
from ecbm4040.classifiers.softmax import *


class BasicClassifier(object):
    def __init__(self):
        self.W = None
        self.velocity = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, optim='SGD', momentum=0.5, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent(SGD).

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
        - y: a numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - optim: the optimization method, the default optimizer is 'SGD' and
                     feel free to add other optimizers.
        - verbose: (boolean) if true, print progress during optimization.

        Outputs:
        - loss_history: a list containing the value of the loss function of each iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        # Initialize W and velocity(for SGD with momentum)
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        if self.velocity is None:
            self.velocity = np.zeros_like(self.W)

        # Run stochastic gradient descent to optimize W
        
        loss_history = []
        
        for it in range(num_iters):

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sometimes, random     #
            # choice will be better than training in order.                         #
            #########################################################################

            X_batch = np.zeros([batch_size, dim])
            y_batch = np.zeros(batch_size)

            batch_index = np.random.choice(np.arange(0, X.shape[0]), batch_size, replace=False)
            
            X_batch = X[batch_index, :]
            y_batch = y[batch_index]
            
            loss, dW = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.W = self.W - learning_rate * dW
            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.

        Returns:
        - y_pred: predicted labels for the data in X. y_pred is a 1-dimensional
                  array of length N, and each element is an integer giving the predicted
                  class.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        
        y_pred = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            pre = np.dot(X[i, :], self.W).tolist()
            y_pred[i] = pre.index(max(pre))

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: a numpy array of shape (N, D) containing a minibatch of N
                  data points; each point has dimension D.
        - y_batch: a numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: a tuple containing:
        - loss:  a single float
        - gradient:  gradients wrt W, an array of the same shape as W
        """
        pass


class LinearSVM(BasicClassifier):
    """ A subclass that uses the Multiclass Linear SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(BasicClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
