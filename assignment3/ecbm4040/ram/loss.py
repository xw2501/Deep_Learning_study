import tensorflow as tf
from ecbm4040.ram.utils import *


########################################################
# Hybrid loss function: reward, cross_entropy,         #
# log-likelihood of location samples, baseline         #          
########################################################
def reward(softmax_a, labels_ph):
    """
    Normally, reward is considered as a feedback signal after the agent
    execute its action and the goal of the agent/network is to 
    maximize the total reward along the whole T steps.
    
    In the case of object recognition/classification, 
    for example, total_reward(T) = 1 if the object is classified 
    correctly after T steps and 0 otherwise. Here T is the number 
    of glimpse.

    :param softmax_a: output of action net, (batch_size, num_classes)
    :param labels_ph: ground truth labels, (batch_size,)

    :return r_T: the total reward after num_glimpse. (batch_size,)
    """
    prediction = tf.argmax(softmax_a, 1)  # (batch_size,)
    r_T = tf.cast(tf.equal(prediction, labels_ph), tf.float32)  # (batch_size,)

    return r_T


def softmax_cross_entropy(a, labels_ph):
    """
    For classification problems, we optimize the cross entropy loss 
    to train the action network fa and back-propagate the gradients
    through the core and glimpse networks.

    :param a: output of action net, with shape (batch_size, num_classes)
    :param labels_ph: ground truth labels, (batch_size,)

    :return cross_entropy: a scalar, average cross_entropy of all samples in a batch
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=a, labels=labels_ph)
    cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy


def loc_loglikelihood(loc_means, loc_samples, loc_std):
    """
    :param loc_means: a list of coordinate means, [(batch_size, loc_dim), (batch_size, loc_dim,), ..., (batch_size, loc_dim,)]
    :param loc_samples: a list of coordinate samples, [(batch_size, loc_dim), (batch_size, loc_dim,), ..., (batch_size, loc_dim,)]
    :param loc_std: a fixed variance defined in config

    :return logll: log likelihood of location, (batch_size, num_glimpses)
    """
    loc_means = tf.stack(loc_means)  # (num_glimpse, batch_size, loc_dim)
    loc_samples = tf.stack(loc_samples)  # (num_glimpse, batch_size, loc_dim)

    gaussian_dist = tf.distributions.Normal(loc=loc_means, scale=loc_std)

    logll = gaussian_dist.log_prob(loc_samples)  # (num_glimpses, batch_size, loc_dim)
    logll = tf.reduce_sum(logll, 2)  # (num_glimpses, batch_size)
    logll = tf.transpose(logll)  # (batch_size, num_glimpses)

    logll = tf.stop_gradient(logll)

    return logll


def baseline_net(outputs, cell_dim):
    # Time independent baselines
    w_baseline = weight_variable((cell_dim, 1))
    b_baseline = bias_variable((1,))

    baselines = []
    for t, output in enumerate(outputs):
        baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
        baseline_t = tf.squeeze(baseline_t)
        baselines.append(baseline_t)

    baselines = tf.stack(baselines)  # (num_glimpses, batch_size)
    baselines = tf.transpose(baselines)  # (batch_size, num_glimpses)

    return baselines
