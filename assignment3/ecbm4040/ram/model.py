import tensorflow as tf

from ecbm4040.ram.networks import *
from ecbm4040.ram.loss import *
from tensorflow.contrib.rnn import LSTMCell


########################################################
#             the core rnn network.                    #          
########################################################
def core_rnn_net(config, images_ph):
    """
    Inputs:
    :param config
    :param images_ph: (batch_size, image_size, image_size, num_channels)
    
    Returns:
    :param outputs: a list/tuple of length num_glimpse with outputs/hidden_states (batch_size, cell_dim) from rnn network
    :param loc_means: a list of length num_glimpse with all output coordinate mean of loc_net
    :param loc_samples: a list of length num_glimpse with all sampled next_loc from loc_net
    """

    cell_dim = config.get("cell_dim", None)
    num_glimpses = config.get("num_glimpses", None)
    loc_dim = config.get("loc_dim", None)
    batch_size = tf.shape(images_ph)[0]

    with tf.variable_scope("glimpse_net", reuse=None):
        glimpse_net = GlimpseNet(config, images_ph)

    with tf.variable_scope("loc_net", reuse=None):
        loc_net = LocNet(config)

        ###############################################
        # TODO: the core rnn network and use LSTM cell#
        ###############################################
        # First, set up initial variables. For example,
        # init_loc: randomly and uniformly sampled
        #           from -1 to 1. You can use "tf.random_uniform" here.
        # init_g: the first input into rnn core network.
        #
        # Then define a LSTM cell by tf.contrib.rnn.LSTMCell as you did in task 1.
        # Also, you need to initialize the state values, and you can use "zero_state"
        # function. Go to https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell
        # and get familiar with this class.
        #
        # Finally, create a RNN with time_steps = num_glimpses
        # Be careful that in this RNN, you need to feed the output next_loc back to the input
        # of the glimpse net.
        # Hint: There are many ways to feed the output back
        # to the input of the rnn network,
        # 1. for loop
        # 2. use tf.contrib.legacy_seq2seq modules and define the
        #    "loop function" to generate next input from the output
        #    of the current step. This method is recommended, since
        #    it will help you get familiar with some advanced RNN modules
        #    provided by tensorflow.
        #    In detail, a "loop function" should look like,
        #    def loop_fn(h_t, t):
        #        loc_mean, next_loc = loc_net(h_t)
        #        next_glimpse = glimpse_net(next_loc)
        #        return next_glimpse
        #
        #    Also, you need to cache the loc_mean and next_loc in a list,
        #    because you are going to use them in loss computation.
        
    _loc = tf.random_uniform(shape=[batch_size, loc_dim], minval=-1, maxval=1)
    cell = LSTMCell(num_units=cell_dim)
    _state = cell.zero_state(batch_size, dtype=tf.float32)
    
    outputs = []
    loc_means = []
    loc_samples = []
    for i in range(num_glimpses):
        _g = glimpse_net(_loc)
        _h, _state = cell(_g, _state)
        _loc_mean, _next_loc = loc_net(_h)
        _loc = _next_loc
        outputs.append(_h)
        loc_means.append(_loc_mean)
        loc_samples.append(_next_loc)

    return outputs, loc_means, loc_samples

    #raise NotImplementedError('Please edit this function.')

    ###############################################
    #               End of your code.             #
    ###############################################


def model(config, train_cfg, reuse_core=False, reuse_action=False):
    """
    Finally, we combine all parts together, including
    the core rnn network, the output action net, loss function
    and the optimizer.
    """

    # placeholders
    images_ph = tf.placeholder(tf.float32, [None, config["image_size"], config["image_size"], config["num_channels"]])
    labels_ph = tf.placeholder(tf.int64, [None])

    # define the networks
    with tf.variable_scope("core_rnn_net", reuse=reuse_core):
        outputs, loc_means, loc_samples = core_rnn_net(config, images_ph)

    with tf.variable_scope("action_net", reuse=reuse_action):
        action_net = ActionNet(config)

    ## baseline_net is for variance reduction
    with tf.variable_scope("baseline_net", reuse=reuse_core):
        b = baseline_net(outputs, config["cell_dim"])

    # get reward of each sample in the batch
    a, softmax_a = action_net(outputs[-1])
    R = tf.expand_dims(reward(softmax_a, labels_ph), 1)  # (batch_size, 1)
    R = tf.tile(R, [1, config["num_glimpses"]])  # (batch_size, num_glimpse)
    R_avg = tf.reduce_mean(R)

    # J: maximization of reward and train core rnn network
    loc_ll = loc_loglikelihood(loc_means, loc_samples, config["loc_std"])
    J = tf.reduce_mean(loc_ll * (R - tf.stop_gradient(b)))  # corresponding to equation (2)

    # make use of prior knowledge to train action nets
    # and also backpropagate back to core_rnn_net
    cross_entropy = softmax_cross_entropy(a, labels_ph)

    # to train baseline net
    baseline_mse = tf.reduce_mean(tf.square(R - b))

    # hybrid loss function
    hybrid_loss = -J + cross_entropy + baseline_mse

    # define the number of correct prediction
    prediction = tf.argmax(softmax_a, 1)  # (batch_size,)
    correct_num = tf.cast(tf.equal(prediction, labels_ph), tf.float32)  # (batch_size,)
    correct_num = tf.reduce_sum(correct_num)

    # back propagation
    params = tf.trainable_variables()
    grads = tf.gradients(hybrid_loss, params)
    grads, _ = tf.clip_by_global_norm(grads, train_cfg["max_grad_norm"])

    # define optimizer
    lr_init = train_cfg["lr_init"]
    lr_min = train_cfg["lr_min"]
    decay_steps = train_cfg["num_train"] // train_cfg["batch_size"]
    decay_rate = train_cfg["decay_rate"]
    ## Optional Variable to increment by one after the variables have been updated.
    global_step = tf.get_variable("global_step", initializer=0, trainable=False)
    lr = tf.train.exponential_decay(lr_init, global_step, decay_steps, decay_rate, staircase=True)
    lr = tf.maximum(lr, lr_min)
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.apply_gradients(zip(grads, params), global_step=global_step)

    return images_ph, labels_ph, hybrid_loss, J, cross_entropy, baseline_mse, R_avg, correct_num, lr, train_step, loc_means, loc_samples
