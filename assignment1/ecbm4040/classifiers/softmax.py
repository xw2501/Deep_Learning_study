import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    K = W.shape[1]
    
    for i in range(N):
        vec = np.dot(X[i, :], W)
        vec -= np.max(vec)
        e_s = np.sum(np.exp(vec[:]))
        for j in range(K):
            dW[:, j] += (np.exp(vec[j])/e_s) * X[i]
        loss -= np.log(np.exp(vec[y[i]])/e_s)
        dW[:, y[i]] -= X[i]
        
    loss /= N
    dW /= N
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    K = W.shape[1]
    
    vec = np.dot(X, W)
    vec = np.exp(vec - np.max(vec, axis=1).reshape(-1, 1))
    
    e_s = np.sum(vec, axis=1)
    vec /= e_s.reshape(-1, 1)
    
    mask = np.zeros([K, N])
    mask[y, np.arange(N)] += 1
    
    dW -= np.dot(mask, X).T
    loss -= np.sum(np.log(vec[np.arange(N), y]))
    dW += np.dot(vec.T, X).T
    
    loss /= N
    dW /= N

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
