#!/usr/bin/env python
# ECBM E4040 Fall 2017 Assignment 2
# This script is intended for task 5 Kaggle competition. Use it however you want.
import zipfile
import os
import glob
import time
import scipy
import csv
import _pickle as pickle
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

def load_data():
    """
    Unpack the CIFAR-10 dataset and load the datasets.
    :param mode: 'train', or 'test', or 'all'. Specify the training set or test set, or load all the data.
    :return: A tuple of data/labels, depending on the chosen mode. If 'train', return training data and labels;
    If 'test' ,return test data and labels; If 'all', return both training and test sets.
    """

    if os.path.exists('./data/kaggle_test_128.zip') and os.path.exists('./data/kaggle_train_128.zip'):
        print('file exists. Begin extracting...')
    else:
        print('file not uploaded')
        return None

    if not os.path.exists('./data/test_128/'):
        package = zipfile.ZipFile('./data/kaggle_test_128.zip')
        package.extractall('./data')
        package.close()
        
    if not os.path.exists('./data/train_128/'):
        package = zipfile.ZipFile('./data/kaggle_train_128.zip')
        package.extractall('./data')
        package.close()

    root_dir = os.getcwd()
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    final_test = []
    for sublabel in range(5):
        os.chdir('./data/train_128/'+str(sublabel))
        data_train = glob.glob('*')
        try:
            for name in data_train:
                image = scipy.misc.imread(name)
                train_data.append(image)
                train_label.append(sublabel)
        except BaseException:
            print('Something went wrong...')
            return None
        os.chdir(root_dir)
        
    os.chdir('./data/test_128')
    for i in range(3500):
        name = str(i)+'.png'
        image = scipy.misc.imread(name)
        final_test.append(image)
    os.chdir(root_dir)
    
    final_test = np.asarray(final_test)
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    
    indice = np.random.choice(np.arange(train_data.shape[0]), int(train_data.shape[0]*0.1), replace = False)
    
    test_data = train_data[indice]
    test_label = train_label[indice]
    train_data = np.delete(train_data, indice, axis=0)
    train_label = np.delete(train_label, indice)
    
    return train_data, train_label, test_data, test_label, final_test
    
    
def compress_data(data):
    compressed = data[:, :, 32:96, :]
    for i in range(64):
        compressed[:, i, :, :] = compressed[:, i*2, :, :]
    compressed = compressed[:, 0:64, :, :]
    return compressed


def variables_vgg16(patch_size1=3, patch_size2=3, patch_size3=3, patch_size4=3, patch_depth1=64, patch_depth2=128, patch_depth3=256, patch_depth4=512, num_hidden1=4096, num_hidden2=1000, image_width=64, image_height=64, image_depth=3, num_labels=5, reg_tf=0.01):
    
    variables = {
        'patch_size1':patch_size1, 'patch_size2':patch_size2, 'patch_size3':patch_size3, 'patch_size4':patch_size4, 'patch_depth1':patch_depth1, 'patch_depth2':patch_depth2, 'patch_depth3':patch_depth3, 'patch_depth4':patch_depth4, 'num_hidden1':num_hidden1, 'num_hidden2':num_hidden2, 'image_width':image_width, 'image_height':image_height, 'image_depth':image_depth, 'num_labels':num_labels, 'reg_tf':reg_tf}
    
    return variables


def build_model(x_tf, y_tf, variables):
    
    conv_layer1 = conv_layer(input_x=x_tf, in_channel=variables['image_depth'], out_channel=variables['patch_depth1'], kernel_shape=variables['patch_size1'], index=1)
    pool_layer1 = max_pooling_layer(input_x=conv_layer1.output(), k_size=2,padding='SAME')
    
    conv_layer2 = conv_layer(input_x=pool_layer1.output(), in_channel=variables['patch_depth1'], out_channel=variables['patch_depth2'], kernel_shape=variables['patch_size2'], index=2)
    pool_layer2 = max_pooling_layer(input_x=conv_layer2.output(), k_size=2,padding='SAME')
    
    conv_layer3 = conv_layer(input_x=pool_layer2.output(), in_channel=variables['patch_depth2'], out_channel=variables['patch_depth3'], kernel_shape=variables['patch_size3'], index=3)
    pool_layer3 = max_pooling_layer(input_x=conv_layer3.output(), k_size=2,padding='SAME')
    
    conv_layer4 = conv_layer(input_x=pool_layer3.output(), in_channel=variables['patch_depth3'], out_channel=variables['patch_depth4'], kernel_shape=variables['patch_size4'], index=4)
    pool_layer4 = max_pooling_layer(input_x=conv_layer4.output(), k_size=2,padding='SAME')

    pool_shape = pool_layer4.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flat_layer = tf.reshape(pool_layer4.output(), shape=[-1, img_vector_length])

    fc_layer1 = fc_layer(input_x=flat_layer, in_size=img_vector_length, out_size=variables['num_hidden1'], activation_function=tf.nn.relu, index=1)
    drop_layer1 = tf.nn.dropout(fc_layer1.output(), 0.5)
    
    fc_layer2 = fc_layer(input_x=drop_layer1, in_size=variables['num_hidden1'], out_size=variables['num_hidden2'], activation_function=tf.nn.relu, index=2)
    drop_layer2 = tf.nn.dropout(fc_layer2.output(), 0.5)
    
    fc_layer3 = fc_layer(input_x=drop_layer2, in_size=variables['num_hidden2'], out_size=variables['num_labels'], activation_function=None, index=3)

    logits = fc_layer3.output()
    
    conv_w = [conv_layer1.weight, conv_layer2.weight, conv_layer3.weight, conv_layer4.weight]
    fc_w = [fc_layer1.weight, fc_layer2.weight, fc_layer3.weight]
    
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])
        label = tf.one_hot(y_tf, variables['num_labels'])
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits), name='cross_entropy')
        loss = tf.add(cross_entropy_loss, variables['reg_tf'] * l2_loss, name='loss')
        
        tf.summary.scalar('myModel_loss', loss)
    
    return logits, loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 5)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, 
                                       epsilon=1e-08, use_locking=False, 
                                       name='Adam').minimize(loss)
        
    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('myModel_error_num', error_num)
    return error_num, pred


class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed=100, index=0):
        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight
                
            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias
                
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)
            self.cell_out = cell_out
            
            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)
            
    def output(self):
        return self.cell_out
    
    
class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        with tf.variable_scope('max_pooling'):
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

                
class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed=100, activation_function=None, index=0):
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)
            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)
            
    def output(self):
        return self.cell_out
    
    
def my_training(X_train, y_train, X_val, y_val, final_data, final_label, variables, seed=100, learning_rate=1e-3, epoch=20, batch_size=500, verbose=False, pre_trained_model=None):
    print("learning_rate={}".format(learning_rate))

    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, variables['image_height'], variables['image_width'], variables['image_depth']], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = build_model(xs, ys, variables)
    
    iters = int(X_train.shape[0] / batch_size)
    step = train_step(loss, learning_rate)
    print('number of batches for training: {}'.format(iters))

    eve, pre = evaluate(output, ys)
    
    iter_total = 0
    best_acc = 0
    cur_model_name = 'myModel_{}'.format(int(time.time()))
    prediction = 0
    vali = 0

    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    writer.add_summary(merge_result, iter_total)
                    
                    # when achieve the best validation accuracy, we store the model paramters
                    print('current accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                        prediction = sess.run(pre, feed_dict={xs: final_data, ys: final_label})
        
        
    print("Traning ends. The best valid accuracy is {}.".format(best_acc))
    
    return prediction