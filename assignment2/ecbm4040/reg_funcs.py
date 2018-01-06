#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This script contains forward/backward functions of regularization techniques

import numpy as np


def dropout_forward(x, dropout_config, mode):
    """
    Dropout feedforward
    :param x: input tensor with shape (N, D)
    :param dropout_config: (dict)
                           enabled: (bool) indicate whether dropout is used.
                           keep_prob: (float) retention rate, usually range from 0.5 to 1 and close to 1
    :param mode: (string) "train" or "test"
    :return:
    - out: a tensor with the same shape as x
    - cache: (train phase) cache a random dropout mask used in feedforward process
             (test phase) None
    """
    keep_prob = dropout_config.get("keep_prob", 0.7)
    
    out, cache = None, None
    if mode == "train":
        ###########################################
        # TODO: Implement training phase dropout. #
        # Remember to return retention mask for   #
        # backward.                               #
        ###########################################
        N = x.shape[0]
        D = x.shape[1]
        cache = np.random.binomial(1, keep_prob, N*D).reshape(N, D)
        out = np.multiply(x, cache)
        
        # raise NotImplementedError
    elif mode == "test":
        ##########################################
        # TODO: Implement test phase dropout. No #
        # need to use mask here.                 #
        ##########################################
        out = x
        
        # raise NotImplementedError
    return out, cache


def dropout_backward(dout, cache):
    """
    Dropout backward only for train phase.
    :param dout: a tensor with shape (N, D)
    :param cache: (tensor) mask, a tensor with the same shape as x
    :return: dx: the gradients transfering to the previous layer
    """
    dx = cache * dout
    return dx


def bn_forward(x, gamma, beta, bn_params, mode):
    """
    Batch Normalization forward
    
    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has D features. We will apply
    mini-batch normalization on N samples in x. 
    
    In the "train" mode:
    1. Apply normalization transform to input x and store in out.
       current_mean, current_var = mean(x), var(x)
       out = gamma*(x-current_mean)/sqrt(current_var+epsilon) + beta
    
    2. Update mean and variance estimation in bn_config using moving average method, ie.,
       moving_mean = decay*moving_mean + (1-decay)*current_mean
       moving_var = decay*moving_var + (1-decay)*current_var
       
    Side note:
    Here we use the moving average strategy to estimiate the mean and var of the data.
    It is kind of approximation to the mean and var of the training data. Also, this is
    a popular strategy and tensorflow use it in their implementation.
    
    In the "test" mode: 
    Instead of using the mean and var of the input data, it is going to use mean and var
    stored in bn_config to make normalization transform.
    
    :param x: a tensor with shape (N, D)
    :param gamma: (tensor) a scale tensor of length D, a trainable parameter in batch normalization.
    :param beta:  (tensor) an offset tensor of length D, a trainable parameter in batch normalization.
    :param bn_params:  (dict) including epsilon, decay, moving_mean, moving_var.
    :param mode:  (string) "train" or "test".
    
    :return:
    - out: a tensor with the same shape as input x.
    - cahce: (tuple) contains (x, gamma, beta, eps, mean, var)
    """
    eps = bn_params.get("epsilon", 1e-5)
    decay = bn_params.get("decay", 0.9)

    N, D = x.shape
    moving_mean = bn_params.get('moving_mean', np.zeros(D, dtype=x.dtype))
    moving_var = bn_params.get('moving_var', np.ones(D, dtype=x.dtype))

    out, mean, var = None, None, None
    if mode == "train":
        #############################################################
        # TODO: Batch normalization forward train mode               #
        #      1. calculate mean and variance of input x            #
        #      2. normalize x with mean and variance                #
        #      3. apply scale(gamma) and offset(beta) on the        #
        #         normalized data                                   #
        #      4. remenber to use moving average method to update   #
        #         moving_mean and moving_var in the bn_params       #
        #############################################################
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        out = np.zeros(x.shape)
        
        moving_mean = decay*moving_mean + (1-decay)*mean
        moving_var = decay*moving_var + (1-decay)*var
        
        for i in range(N):
            out[i, :] = gamma*(x[i, :]-mean)/np.power(var+eps, 0.5) + beta
            
        # raise NotImplementedError
    elif mode == 'test':
        #######################################################################
        # TODO: Batch normalization forward test mode                         #
        #######################################################################
        mean = moving_mean
        var = moving_var
        out = np.zeros(x.shape)
        
        for i in range(N):
            out[i, :] = gamma*(x[i, :]-mean)/np.power(var+eps, 0.5) + beta
        
        # raise NotImplementedError

    # cache for back-propagation
    cache = (x, gamma, beta, eps, mean, var)
    # Update mean and variance estimation in bn_config
    bn_params['moving_mean'] = moving_mean
    bn_params['moving_var'] = moving_var

    return out, cache


def bn_backward(dout, cache):
    """
    Batch normalization backward
    Derive the gradients wrt gamma, beta and x

    :param dout:  a tensor with shape (N, D)
    :param cache:  (tuple) contains (x, gamma, beta, eps, mean, var)
    
    :return:
    - dx, dgamma, dbeta
    """
    x, gamma, beta, eps, mean, var = cache
    N, D = dout.shape

    dx, dgamma, dbeta = None, None, None
    x_hat = (x - mean) / np.sqrt(np.tile(var, (N, 1)) + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dout * gamma / np.sqrt(np.tile(var, (N, 1)) + eps)
    return dx, dgamma, dbeta
