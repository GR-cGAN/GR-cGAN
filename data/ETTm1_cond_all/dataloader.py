import numpy as np
import os
import pandas as pd
import data.data_utils as data_utils


def data_loading(params):
    """Load and preprocess real-world datasets.

    Args:
        - data_name: stock or energy
        - seq_len: sequence length

    Returns:
        - data: preprocessed data.
    """
    seq_len = params.seq_len
    stride_size = params.stride_size
    ori_data = pd.read_csv(os.path.join(params.raw_data_path, params.name), index_col='date', parse_dates=True)
    ori_data = ori_data.to_numpy()

    # Normalize the data
    ori_data = data_utils.MinMaxScaler(ori_data)

    # Preprocess the dataset
    data = np.zeros(((len(ori_data) - seq_len) // stride_size, seq_len, ori_data.shape[1]))
    # Cut data by sequence length
    for i in range(data.shape[0]):
        j = i * stride_size
        data[i] = ori_data[j:j + seq_len]

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(data))
    data = data[idx]

    train_data = data

    return train_data, data
