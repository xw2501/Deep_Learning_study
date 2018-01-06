#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

def my_sigmoid(x):
    output = []
    for i in range(x.shape[0]):
        output.append(1/(1+np.exp(-x[i])))
            
    return output

def my_tanh(x):
    output = []
    for i in range(x.shape[0]):
        output.append(np.tanh(x[i]))
            
    return output

class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        params = []
        params.append(num_units) # number of units
        params.append(num_proj)  # output size
        params.append(forget_bias)
        params.append(activation)
        self.params = params
        
        W_fh = tf.Variable(tf.random_normal([num_proj, 1]), name='W_fh', dtype=tf.float32)
        W_ih = tf.Variable(tf.random_normal([num_proj, 1]), name='W_ih', dtype=tf.float32)
        W_ch = tf.Variable(tf.random_normal([num_proj, 1]), name='W_ch', dtype=tf.float32)
        W_oh = tf.Variable(tf.random_normal([num_proj, 1]), name='W_oh', dtype=tf.float32)
        W_fc = tf.Variable(tf.random_normal([num_units, 1]), name='W_fc', dtype=tf.float32)
        W_ic = tf.Variable(tf.random_normal([num_units, 1]), name='W_ic', dtype=tf.float32)
        W_oc = tf.Variable(tf.random_normal([num_units, 1]), name='W_oc', dtype=tf.float32)
        W_fi = tf.Variable(tf.random_normal([1, 1]), name='W_fi', dtype=tf.float32)
        W_ii = tf.Variable(tf.random_normal([1, 1]), name='W_ii', dtype=tf.float32)
        W_ci = tf.Variable(tf.random_normal([1, 1]), name='W_ci', dtype=tf.float32)
        W_oi = tf.Variable(tf.random_normal([1, 1]), name='W_oi', dtype=tf.float32)
        W_h = tf.Variable(tf.random_normal([num_units, num_proj]), name='W_h', dtype=tf.float32)
        
        b_f = tf.Variable(tf.random_normal([1, 1]), name='b_f', dtype=tf.float32)
        b_i = tf.Variable(tf.random_normal([1, 1]), name='b_i', dtype=tf.float32)
        b_c = tf.Variable(tf.random_normal([1, 1]), name='b_c', dtype=tf.float32)
        b_o = tf.Variable(tf.random_normal([1, 1]), name='b_o', dtype=tf.float32)
        
        W = {
            'W_fh':W_fh, 'W_ih':W_ih, 'W_ch':W_ch, 'W_oh':W_oh, 'W_fi':W_fi, 'W_ii':W_ii, 'W_ci':W_ci, 'W_oi':W_oi, 'W_h':W_h, 'W_fc':W_fc, 'W_ic':W_ic, 'W_oc':W_oc
        }
        
        b = {
            'b_f':b_f, 'b_i':b_i, 'b_c':b_c, 'b_o':b_o
        }
        
        self.W = W
        self.b = b
        
        #raise NotImplementedError('Please edit this function.')

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        params = self.params
        return params[0]+params[1]
        
        #raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        params = self.params
        return params[1]
        
        #raise NotImplementedError('Please edit this function.')

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        params = self.params
        
        c_prev = array_ops.slice(state, [0, 0], [-1, params[0]])
        h_prev = array_ops.slice(state, [0, params[0]], [-1, params[1]])
        
        W = self.W
        b = self.b

        W_fh = W['W_fh']
        W_ih = W['W_ih']
        W_ch = W['W_ch']
        W_oh = W['W_oh']
        W_fi = W['W_fi']
        W_ii = W['W_ii']
        W_ci = W['W_ci']
        W_oi = W['W_oi']
        W_h = W['W_h']
        W_fc = W['W_fc']
        W_ic = W['W_ic']
        W_oc = W['W_oc']
            
        b_f = b['b_f']
        b_i = b['b_i']
        b_c = b['b_c']
        b_o = b['b_o']
            
        f = math_ops.sigmoid(tf.matmul(h_prev, W_fh) + tf.multiply(inputs, W_fi) + b_f + tf.matmul(c_prev, W_fc))
        i = math_ops.sigmoid(tf.matmul(h_prev, W_ih) + tf.multiply(inputs, W_ii) + b_i + tf.matmul(c_prev, W_ic))
        _c = math_ops.tanh(tf.matmul(h_prev, W_ch) + tf.multiply(inputs, W_ci) + b_c)
        c = f * c_prev + i * _c
        o = math_ops.sigmoid(tf.matmul(h_prev, W_oh) + tf.multiply(inputs, W_oi) + b_o + tf.matmul(c, W_oc))
            
        h = o * math_ops.tanh(c)
        h = tf.matmul(h, W_h)

        new_state = (array_ops.concat([c, h], 1))
        output = h
        
        return output, new_state
        #raise NotImplementedError('Please edit this function.')
        
