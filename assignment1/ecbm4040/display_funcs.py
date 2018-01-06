import numpy as np
import matplotlib.pyplot as plt

def visualize_pics(pics):
    ## visualization
    num_pics = pics.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_pics)))
    padding = 2
    figure = np.zeros(((32+2*padding)*grid_size, (32+2*padding)*grid_size, 3))
    for r in range(grid_size):
        for c in range(grid_size):
            pid = r*grid_size + c
            if pid < num_pics:
                pic = pics[pid]
                high, low = np.max(pic), np.min(pic)
                pic = 255.0*(pic-low)/(high-low)
                rid = (32+2*padding)*r
                cid = (32+2*padding)*c
                figure[rid+padding:rid+padding+32, cid+padding:cid+padding+32, :] = pic

    print('num of feature vectors: {}'.format(num_pics))
    plt.imshow(figure.astype('uint8'))
    plt.gca().axis('off')
    plt.show()