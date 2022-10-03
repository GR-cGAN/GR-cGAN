import numpy as np
import os
import data.data_utils as data_utils
import pathlib


def data_loading(params):
    """Load and preprocess real-world datasets.

    Args:
        - data_name: stock or energy
        - seq_len: sequence length

    Returns:
        - data: preprocessed data.
    """
    seq_len = params.seq_len

    pathlib.Path().resolve()
    ori_data = np.loadtxt(os.path.join(params.raw_data_path, params.name), delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = data_utils.MinMaxScaler(ori_data)

    # Preprocess the dataset
    data = np.zeros((len(ori_data) - seq_len, seq_len, ori_data.shape[1]))
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        data[i] = ori_data[i:i + seq_len]

    data = np.float32(data)
    train_data = data[:int(data.shape[0] * 0.7)]

    return train_data, data
