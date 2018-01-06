import numpy as np


def create_dataset(num_samples, seq_len=8):
    """
    This function creates a dataset needed for XOR network. It outputs the input(data) and the corresponding tag(output)
    of the XOR network.

    :param num_samples: The total number of samples you would like for training.
    :param seq_len: The length of each training input. This determines the second dimension of data.

    :return data: A randomly generated numpy matrix with size [num_samples, seq_len] that only contains 0 and 1.
    :return output: A numpy matrix with size [num_samples, seq_len]. The value of this matrix follows:
                    output[i][j] = data[i][0] ^ data[i][1] ^ data[i][2] ^ ... ^ data[i][j]

    """
    data = np.random.randint(2, size=(num_samples, seq_len, 1))

    output = np.zeros([num_samples, seq_len], dtype=np.int)
    for sample, out in zip(data, output):
        count = 0
        for c, bit in enumerate(sample):
            if bit[0] == 1:
                count += 1
            out[c] = 1 - int(count % 2 == 0)
    return data, output
