import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize


######################################
#       tensorflow utilities         #
######################################
def weight_variable(shape, name=None):
    """
    :param shape: a list/tuple of integers
    """
    if name == None:
        name = "weights"
    init = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.get_variable(name, initializer=init)


def bias_variable(shape, name=None):
    """
    :param shape: a list/tuple of integers
    """
    if name == None:
        name = "bias"
    init = tf.zeros(shape, tf.float32)
    return tf.get_variable(name, initializer=init)


######################################
#           image utilities          #
######################################
def translate_60_mnist(images, image_size=28, num_channels=1):
    # output 60x60 translated images
    batch_size = images.shape[0]
    images = images.reshape((batch_size, image_size, image_size, num_channels))

    rx, ry = np.random.randint(0, 60 - image_size), np.random.randint(0, 60 - image_size)
    images = np.lib.pad(images, ((0, 0), (rx, 60 - image_size - rx), (ry, 60 - image_size - ry), (0, 0)), "constant")

    return images  # (batch_size, 60, 60, num_channel)


def mnist_addition_pair(images, labels, image_size=28, num_channels=1):
    # output 60x60 images with two-digit
    batch_size = images.shape[0]
    images = images.reshape((batch_size, image_size, image_size, num_channels))

    image_pairs = []
    pair_labels = []
    for i in range(batch_size):
        lx, ly = np.random.randint(0, 60 - image_size), np.random.randint(0, 30 - image_size)
        rx, ry = np.random.randint(0, 60 - image_size), np.random.randint(30, 60 - image_size)
        image_pair = np.zeros((60, 60, 1))
        image_pair[lx:lx + 28, ly:ly + 28, :] = images[i]
        image_pair[rx:rx + 28, ry:ry + 28, :] = images[batch_size - 1 - i]
        image_pairs.append(image_pair)
        pair_labels.append(labels[i] + labels[batch_size - 1 - i])

    # stack image pairs along a new axis    
    image_pairs = np.stack(image_pairs, 0)  # [batch_size, image_size, image_size, 1]
    pair_labels = np.stack(pair_labels, 0)  # [batch_size,]

    return image_pairs, pair_labels


# visualize result
def glimpse_path(images, paths, original_win, scale):
    ## visualization of attention path
    paths = np.stack(paths, 0)  # (num_glimpse, batch_size, 2)
    paths = np.transpose(paths, (1, 0, 2))  # (batch_size, num_glimpse, 2)
    num_imgs = paths.shape[0]
    image_size = images.shape[1]
    num_glimpse = paths.shape[1]

    f, axarr = plt.subplots(num_imgs, num_glimpse + 1, figsize=(4 * (num_glimpse + 1), 4 * num_imgs))
    for i in range(num_imgs):
        # show path
        axarr[i][0].imshow(np.squeeze(images[i]), cmap="gray")
        axarr[i][0].plot(image_size * (paths[0, :, 1] + 1) / 2., image_size * (paths[0, :, 0] + 1) / 2., "r-")
        # show glimpse patches
        for j in range(1, num_glimpse + 1):
            img = np.squeeze(images[i])
            temp = np.zeros_like(img)

            for s in range(scale):
                win = original_win * (2 ** s)
                x, y = int(image_size * (paths[0, j - 1, 1] + 1) / 2.), int(image_size * (paths[0, j - 1, 0] + 1) / 2.)
                xmin = max(0, x - win // 2)
                ymin = max(0, y - win // 2)
                xmax = min(image_size, x + win // 2)
                ymax = min(image_size, y + win // 2)
                patch = img[ymin:ymax, xmin:xmax]
                patch = imresize(patch, [original_win, original_win])
                patch = imresize(patch, [ymax - ymin, xmax - xmin])
                temp[ymin:ymax, xmin:xmax] += patch

            axarr[i][j].imshow(temp, cmap="gray")

    plt.show()
