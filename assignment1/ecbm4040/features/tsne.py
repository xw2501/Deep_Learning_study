import time
import numpy as np

def p_entropy(X2, sigma):

    P = np.exp(-X2*sigma)
    P = P / np.sum(P)
    #print(np.sum(-X2*sigma))
    H = -np.dot(np.log(P), P)

    return P,H

def binary_search(X, perplexity):

    tol = 1e-5
    goal = np.log(perplexity)
    N, D = X.shape
    sigma = np.ones((N,)).astype('float')
    P = np.zeros((N,N))

    # X2: norm2 distance matrix - NxN
    X2 = (-2*np.dot(X, X.T) + np.sum(X**2, axis=1)).T + np.sum(X**2, axis=1)

    # loop over all points
    for i in range(N):

        sigma_max = np.inf
        sigma_min = 0
        maxiter = 50

        sigma_i = sigma[i]

        for t in range(maxiter):
            X2_i = X2[i, np.concatenate((np.r_[0:i], np.r_[i+1:N]))]
            Pi, Hi = p_entropy(X2_i, sigma_i)
            # binary search for a correct sigma_i
            if abs(Hi-goal) < tol:
                break
            else:
                if Hi > goal:
                    sigma_min = sigma_i
                    if sigma_max == np.inf:
                        sigma_i *= 2
                    else:
                        sigma_i = (sigma_i + sigma_max)/2
                else:
                    sigma_max = sigma_i
                    sigma_i = (sigma_i + sigma_min)/2

        # Set Pi
        sigma[i] = sigma_i
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:N]))] = Pi

    #Return P
    print(np.mean(np.sqrt(1/sigma)))
    return P 
    

def tsne(X, low_dim=2, perplexity=30.0):
    """
    tSNE

    Inputs:
    - X: (float) an array of shape(N,D)
    - low_dim: (int) dimenional of output data
    - pca_dim: (int) rather than using the raw data, we can apply
                pca preprocessing to reduce the dimension to pca_dim
    - perplexity:

    Returns;
    - Y: (float) an array of shape (N,low_dim)
    """
    
    N,D = X.shape

    P = binary_search(X, perplexity)
    P = (P + P.T) / (2*N)
    P = np.maximum(P, 1e-12)
    P *= 4
    Y = np.random.normal(0, 1e-4, (N,low_dim))

    T = 1000
    # training parameters
    momentum = 0.5 # initial momentum
    V = np.zeros_like(Y)
    lr = 100 # initial learning rate
    beta = 0.8
    kappa = 0.2
    gamma = 0.2
    mu = np.ones_like(V)

    tic = time.time()
    for t in range(T):
        
        Y2 = (-2*np.dot(Y, Y.T) + np.sum(Y**2, axis=1)).T + np.sum(Y**2, axis=1)
        Q_numerator = 1/(1 + Y2)
        Q = Q_numerator
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q,1e-12)

        #tic = time.time()
        dY = np.zeros_like(Y)
        for i in range(N):
            dY[i,:] = 4*np.dot((P[i,:]-Q[i,:])*Q_numerator[i,:], Y-Y[i,:])
        #print("loop time: {}".format(time.time()-tic))
        
        # calculate learning rate
        dY_hat = (1-beta)*dY + beta*V
        mu[np.where(dY*dY_hat>0)] = mu[np.where(dY*dY_hat>0)] + kappa
        mu[np.where(dY*dY_hat<0)] = (1-gamma)*mu[np.where(dY*dY_hat<0)]
        #mu[np.where(dY*V==0)] = mu[np.where(dY*V==0)]
        # update
        if t > 250:
            momentum = 0.8
        V = momentum*V + lr*mu*dY
        Y += V

        # stop early exaggeration
        if t == 100:
            P /= 4

        # verbose: report intermediate result
        if (t+1) % 100 == 0:
            cost = np.sum(P * np.log(P / Q));
            print('The {} th loop cost: {}, computation time: {}'.format(t+1, cost, time.time()-tic))
        
    return Y
    
