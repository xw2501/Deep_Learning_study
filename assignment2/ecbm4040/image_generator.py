#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to
        # False.

        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        num = x.shape[0]
        height = 32
        width = 32
        pixels_translated = [0, 0]
        channels = 3
        degree = 0
        is_horizontal_flip = False
        is_vertical_flip = False
        is_add_noise = False
        
        params = {
            'num': num,
            'pixels_translated': pixels_translated,
            'height': height,
            'width': width,
            'degree': degree,
            'is_horizontal_flip': is_horizontal_flip,
            'is_vertical_flip': is_vertical_flip,
            'is_add_noise': is_add_noise
        }
        
        self.x = x.reshape([num,channels,height,width]).transpose((0,2,3,1))
        self.y = y
        self.params = params

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        count = 0
        params = self.params
        x = self.x
        x = x.transpose((0, 3, 1, 2)).reshape([x.shape[0], 32, 32, 3])
        y = self.y
        
        max_batch_num = np.floor(params['num']/batch_size)
        
        while True:
            if count<max_batch_num:
                yield (x[count*batch_size:(count+1)*batch_size], y[count*batch_size:(count+1)*batch_size])
                count = count + 1
            else:
                choice = np.random.choice(np.arange(3))
                print(choice)
                x = self.x
                x = x.transpose((0, 3, 1, 2)).reshape([x.shape[0], 32, 32, 3])
                if choice==1:
                    dis_set = [-1, 0, 1]
                    x = np.roll(x, np.random.choice(dis_set), axis=1)
                    x = np.roll(x, np.random.choice(dis_set), axis=2)
                if choice==0:
                    flip_choice = 0
                    if flip_choice==0:
                        x = np.flip(x, axis=2)
                    if flip_choice==1:
                        x = np.flip(x, axis=1)
                    if flip_choice==2:
                        x = np.flip(x, axis=2)
                        x = np.flip(x, axis=1)
                if choice==3:
                    deg_set = [-1, 0, 1]
                    x = rotate(x, np.random.choice(deg_set), axes=(1, 2))
                    x = x[:, 0:32, 0:32, :]                   

                y = self.y
                rseed = np.random.choice(np.arange(100))
                np.random.seed(rseed)
                np.random.shuffle(x)
                np.random.seed(rseed)
                np.random.shuffle(y)
                count = 0

    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        x = self.x
        
        r = 4
        f, axarr = plt.subplots(r, r, figsize=(8,8))
        for i in range(r):
            for j in range(r):
                img = x[r*i+j]
                axarr[i][j].imshow(img, cmap="gray")

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return:
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels translated.
        # Note: You may wonder what values to append to the edge after the translation. Here, use rolling instead. For
        # example, if you translate 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll
        # (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        params = self.params
        x = self.x
        
        params['pixels_translated'][0] += shift_height
        params['pixels_translated'][1] += shift_width
        
        x = np.roll(x, int(shift_height), axis=1)
        x = np.roll(x, int(shift_width), axis=2)
        
        self.x = x

    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.

        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        params = self.params
        x = self.x
        
        params['degree'] += angle
        
        x = rotate(x, angle, axes=(1, 2))
        
        self.x = x

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        x = self.x
        params = self.params
        
        if mode=='h':
            params['is_horizontal_flip'] = not params['is_horizontal_flip']
            x = np.flip(x, axis=2)
        if mode=='v':
            params['is_vertical_flip'] = not params['is_vertical_flip']
            x = np.flip(x, axis=1)
        if mode=='hv':
            params['is_horizontal_flip'] = not params['is_horizontal_flip']
            x = np.flip(x, axis=2)
            params['is_vertical_flip'] = not params['is_vertical_flip']
            x = np.flip(x, axis=1)
            
        self.x = x

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        x = self.x.transpose((0,3,1,2)).astype('float32')
        params = self.params
        
        params['is_add_noise'] = True
        index = np.random.choice(np.arange(x.shape[0]), int(x.shape[0]*portion), replace=False)
        
        for i in index:
            for k in range(3):
                x[i][k] += np.random.normal(0, amplitude, x[i][k].shape)
            
        self.x = x.transpose((0,2,3,1)).astype('uint8')
        
        
    def x_reset(self):
        params = self.params
        
        self.translate(-params['pixels_translated'][0], -params['pixels_translated'][1])
        self.rotate(-params['degree'])
        if params['is_horizontal_flip']:   
            self.flip('h')
        if params['is_vertical_flip']:   
            self.flip('v')
        
        

