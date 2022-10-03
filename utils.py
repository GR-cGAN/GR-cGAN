"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

(1) train_test_divide: Divide train and test data for both original and synthetic data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) random_generator: random vector generator
(4) batch_generator: mini-batch generator
"""

import json
import logging
import os
import shutil
import sys
import numpy as np
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
# matplotlib.rcParams['savefig.dpi'] = 300  # Uncomment for higher plot resolutions
import matplotlib.pyplot as plt

logger = logging.getLogger('GAN.Utils')


class Params:
    """
    Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __contains__(self, item):
        return item in self.__dict__

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """
    _logger = logging.getLogger('GAN')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%m/%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)
            self.setStream(tqdm)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
    handler = logging.StreamHandler(stream=sys.stdout)
    _logger.addHandler(handler)

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python?noredirect=1&lq=1
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            _logger.info('=*=*=*= Keyboard interrupt =*=*=*=')
            return

        _logger.error("Exception --->", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, itt, checkpoint, ins_name=None, is_best=False):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        itt: (int) number of training iterations
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (str) instance name
    """
    if ins_name is None:
        filepath = os.path.join(checkpoint, f'itt_{itt}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'itt_{itt}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
  
    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = data_x.shape[0]
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = data_x[train_idx]
    test_x = data_x[test_idx]
    train_t = data_t[train_idx]
    test_t = data_t[test_idx]

    # Divide train/test index (synthetic data)
    no = data_x_hat.shape[0]
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = data_x_hat[train_idx]
    test_x_hat = data_x_hat[test_idx]
    train_t_hat = data_t_hat[train_idx]
    test_t_hat = data_t_hat[test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
    time = np.zeros((data.shape[0]))
    max_seq_len = 0
    for i in range(data.shape[0]):
        max_seq_len = max(max_seq_len, data[i, :, 0].shape[0])
        time[i] = data[i, :, 0].shape[0]

    return time, max_seq_len


def random_generator(batch_size, z_dim, max_seq_len):
    """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
    Z_mb = list()
    for i in range(batch_size):
        temp_Z = np.random.uniform(0., 1, [max_seq_len, z_dim])
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, batch_size):
    """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
    no = data.shape[0]
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    x_mb = data[train_idx]
    return x_mb


class TS_sampler(torch.utils.data.Dataset):
    def __init__(self, data):
        super(TS_sampler).__init__()
        self.data = np.float32(data)
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class TS_predictor_sampler(torch.utils.data.Dataset):
    def __init__(self, data, predict_steps):
        super(TS_predictor_sampler).__init__()
        data = np.float32(data)
        self.context = data[:, :-predict_steps]
        self.horizon = data[:, -predict_steps:]
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.context[index], self.horizon[index]

    def __len__(self):
        return self.len


def plot_samples_vae(plot_dir, plot_num, x_mb, x_hat_mb, mu, log_var, z, gen_data, params):
    """
    Plot all chosen samples
    Args:
        plot_dir: path to save plot
        plot_num:
        x_mb:
        x_tilde_mb:
        x_hat_mb:
        gen_data:
        params:
        swd_params:
    :return:

    """

    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    if params.input_dim > len(color_list):
        num_colors = len(color_list)
        x_mb = x_mb[:, :, :num_colors]
        x_hat_mb = x_hat_mb[:, :, :num_colors]
        gen_data = gen_data[:, :, :num_colors]
        plot_dim = num_colors
    else:
        plot_dim = params.input_dim
    x = np.arange(params.seq_len)
    f = plt.figure(figsize=(12, 21), constrained_layout=True)
    nrows = 9
    ncols = 3
    ax = f.subplots(nrows, ncols)
    f.suptitle(f'{params.plot_title}/it_{plot_num}')

    for k in range(nrows):
        if k == 4:
            ax[k, 0].plot(x, x, color='g')
            ax[k, 0].plot(x, x[::-1], color='g')
            ax[k, 0].set_title('This separates real (above) and generated (below) samples', fontsize=8)
        elif k < 4:
            for h in range(plot_dim):
                ax[k, 0].plot(x, x_mb[k, :, h], color=color_list[h], label=params.dim_names[h])
                ax[k, 1].plot(x, x_hat_mb[k, :, h], color=color_list[h], label=params.dim_names[h])
            ax[k, 0].set_title(f'Original data: mu: {mu[k, 0]:.3f}, {mu[k, 1]:.3f}, {mu[k, 2]:.3f}, {mu[k, 3]:.3f}\n'
                               f'log_var: {log_var[k, 0]:.3f}, {log_var[k, 1]:.3f}, {log_var[k, 2]:.3f}, {log_var[k, 3]:.3f}')
            ax[k, 1].set_title('Recovered data: x_hat')
            ax[k, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
            ax[k, 1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
        else:
            for i in range(ncols):
                gen_index = 3 * (k - 5) + i
                for h in range(plot_dim):
                    ax[k, i].plot(x, gen_data[gen_index, :, h], color=color_list[h])
                ax[k, i].set_title(
                    f'z: {z[gen_index, 0]:.3f}, {z[gen_index, 1]:.3f}, {z[gen_index, 2]:.3f}, {z[gen_index, 3]:.3f}')

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def plot_condition_samples(plot_dir, plot_num, x_mb, x_tilde_mb, x_hat_mb, gen_data, cond_data, params, condition_len):
    """
    Plot all chosen samples
    Args:
        plot_dir: path to save plot
        plot_num:
        x_mb:
        x_tilde_mb:
        x_hat_mb:
        gen_data:
        params:
        swd_params:
    :return:

    """

    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    if params.input_dim > len(color_list):
        num_colors = len(color_list)
        x_mb = x_mb[:, :, :num_colors]
        x_tilde_mb = x_tilde_mb[:, :, :num_colors]
        x_hat_mb = x_hat_mb[:, :, :num_colors]
        gen_data = gen_data[:, :, :num_colors]
        plot_dim = num_colors
    else:
        plot_dim = params.input_dim
    x = np.arange(params.seq_len)
    f = plt.figure(figsize=(12, 21), constrained_layout=True)
    nrows = 9
    ncols = 3
    ax = f.subplots(nrows, ncols)
    f.suptitle(f'{params.plot_title}/it_{plot_num}')

    for k in range(nrows):
        if k == 4:
            ax[k, 0].plot(x, x, color='g')
            ax[k, 0].plot(x, x[::-1], color='g')
            ax[k, 0].set_title('This separates real (above) and generated (below) samples', fontsize=8)
        elif k < 4:
            for h in range(plot_dim):
                ax[k, 0].plot(x, x_mb[k, :, h], color=color_list[h])
                ax[k, 1].plot(x, x_tilde_mb[k, :, h], color=color_list[h])
                ax[k, 2].plot(x, x_hat_mb[k, :, h], color=color_list[h])
            else:
                ax[k, 0].set_title('Original data: x')
                ax[k, 1].set_title('Recovered data: x_tilde')
                ax[k, 2].set_title('Recovered next-step data: x_hat')
        else:
            if k == 5:
                for i in range(ncols):
                    ax[k, i].set_title('Generated data on seen conditions')
                    ax[k+1, i].set_title('Raw condition + gt')
                    for h in range(plot_dim):
                        ax[k, i].plot(x, gen_data[i, :, h], color=color_list[h])
                        ax[k, i].axvline(condition_len, color='g', linestyle='dashed', alpha=0.7)
                        ax[k + 1, i].plot(x, cond_data[i, :, h], color=color_list[h])
                        ax[k + 1, i].axvline(condition_len, color='g', linestyle='dashed', alpha=0.7)
            if k == 7:
                for i in range(ncols):
                    ax[k, i].set_title('Generated data on unseen conditions')
                    ax[k+1, i].set_title('Raw condition + gt')
                    for h in range(plot_dim):
                        ax[k, i].plot(x, gen_data[-i-1, :, h], color=color_list[h])
                        ax[k, i].axvline(condition_len, color='g', linestyle='dashed', alpha=0.7)
                        ax[k + 1, i].plot(x, cond_data[-i-1, :, h], color=color_list[h])
                        ax[k + 1, i].axvline(condition_len, color='g', linestyle='dashed', alpha=0.7)

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def plot_samples(plot_dir, plot_num, x_mb, x_tilde_mb, x_hat_mb, gen_data, params, plot_swd=False):
    """
    Plot all chosen samples
    Args:
        plot_dir: path to save plot
        plot_num:
        x_mb:
        x_tilde_mb:
        x_hat_mb:
        gen_data:
        params:
        swd_params:
    :return:

    """

    if plot_swd is True:
        s0 = params.model_param.s0
        c1 = params.model_param.c1
        c2 = params.model_param.c2
        c3 = params.model_param.c3
        init_inv = params.model_param.init_inv
    else:
        s0, c1, c2, c3, init_inv = 0, 0, 0, 0, 0

    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    if params.input_dim > len(color_list):
        num_colors = len(color_list)
        x_mb = x_mb[:, :, :num_colors]
        x_tilde_mb = x_tilde_mb[:, :, :num_colors]
        x_hat_mb = x_hat_mb[:, :, :num_colors]
        gen_data = gen_data[:, :, :num_colors]
        plot_dim = num_colors
    else:
        plot_dim = params.input_dim
    x = np.arange(params.seq_len)
    f = plt.figure(figsize=(24, 42), constrained_layout=True)
    nrows = 9
    ncols = 3
    ax = f.subplots(nrows, ncols)
    f.suptitle(f'{params.plot_title}/it_{plot_num}')

    for k in range(nrows):
        if k == 4:
            ax[k, 0].plot(x, x, color='g')
            ax[k, 0].plot(x, x[::-1], color='g')
            ax[k, 0].set_title('This separates real (above) and generated (below) samples', fontsize=10)
        elif k < 4:
            for h in range(plot_dim):
                ax[k, 0].plot(x, x_mb[k, :, h], color=color_list[h], label=params.dim_names[h])
                ax[k, 1].plot(x, x_tilde_mb[k, :, h], color=color_list[h], label=params.dim_names[h])
                ax[k, 2].plot(x, x_hat_mb[k, :, h], color=color_list[h], label=params.dim_names[h])
            if plot_swd is True:
                ax[k, 0].set_title(f'Original data: x - swd: {price(x_mb[k, :, 0], s0, c1, c2, c3, init_inv):.4f}')
                ax[k, 1].set_title(
                    f'Recovered data: x_tilde - swd: {price(x_tilde_mb[k, :, 0], s0, c1, c2, c3, init_inv):.4f}')
                ax[k, 2].set_title(
                    f'Recovered next-step data: x_hat - swd: {price(x_hat_mb[k, :, 0], s0, c1, c2, c3, init_inv):.4f}')
            else:
                ax[k, 0].set_title('Original data: x')
                ax[k, 1].set_title('Recovered data: x_tilde')
                ax[k, 2].set_title('Recovered next-step data: x_hat')
            ax[k, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
            ax[k, 1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
            ax[k, 2].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
        else:
            for i in range(ncols):
                for h in range(plot_dim):
                    ax[k, i].plot(x, gen_data[3 * (k - 5) + i, :, h], color=color_list[h], label=params.dim_names[h])
                ax[k, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
                if plot_swd is True:
                    ax[k, i].set_title(
                        f'Generated data - swd: {price(gen_data[3 * (k - 5) + i, :, 0], s0, c1, c2, c3, init_inv):.4f}')

    f.savefig(os.path.join(plot_dir, str(plot_num) + '.png'))
    plt.close()


def plot_predictive(plot_dir, judge, plot_num, gt, preds, params):
    predict_steps = preds.shape[1]
    seq_len = gt.shape[1]
    pred_size = preds[0].size
    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    if params.input_dim >= len(color_list):
        num_colors = len(color_list) - 1
        gt = gt[:, :, :num_colors]
        preds = preds[:, :, :num_colors]
        plot_dim = num_colors
    else:
        plot_dim = params.input_dim
    x = np.arange(seq_len)
    f = plt.figure(figsize=(10, 25), constrained_layout=True)
    nrows = gt.shape[0]
    ncols = 2
    ax = f.subplots(nrows, ncols)
    f.suptitle(f'{params.plot_title}/{judge}/it_{plot_num}')

    for k in range(nrows):
        for h in range(plot_dim):
            ax[k, 0].plot(x, gt[k, :, h], color=color_list[h])
            ax[k, 0].axvline(seq_len - predict_steps, color='g', linestyle='dashed', alpha=0.7)
            ax[k, 1].axvline(seq_len - predict_steps, color='g', linestyle='dashed', alpha=0.7)
            ax[k, 1].plot(x[:-predict_steps], gt[k, :-predict_steps, h], color=color_list[h])
            ax[k, 1].plot(x[-predict_steps:], preds[k, -predict_steps:, h], color=color_list[h])
        mae_score = np.sum(np.abs(gt[k, -predict_steps:] - preds[k])) / pred_size
        ax[k, 1].set_title(f'Prediction MAE score: {mae_score:.4f}')

    f.savefig(os.path.join(plot_dir, f'pred_{plot_num}.png'))
    plt.close()


def plot_discriminative(plot_dir, judge, plot_num, fake_data, real_data, fake_score, real_score, params):
    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    if params.input_dim >= len(color_list):
        num_colors = len(color_list) - 1
        fake_data = fake_data[:, :, :num_colors]
        real_data = real_data[:, :, :num_colors]
        plot_dim = num_colors
    else:
        plot_dim = params.input_dim
    x = np.arange(params.seq_len)
    f = plt.figure(figsize=(16, 42), constrained_layout=True)
    nrows = 9
    ncols = 2
    ax = f.subplots(nrows, ncols)
    f.suptitle(f'{params.plot_title}/{judge}/it_{plot_num}')

    for k in range(nrows):
        if k == 4:
            ax[k, 0].plot(x, x, color='g')
            ax[k, 0].plot(x, x[::-1], color='g')
            ax[k, 0].set_title('This separates real (above) and generated (below) samples', fontsize=10)
        elif k < 4:
            for i in range(ncols):
                for h in range(plot_dim):
                    ax[k, i].plot(x, real_data[ncols * k + i, :, h], color=color_list[h], label=params.dim_names[h])
                ax[k, i].plot(x, real_score[ncols * k + i], color=color_list[plot_dim], label='disc_score')
                ax[k, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
                ax[k, i].set_title(f'Original data - score: {np.mean(real_score[ncols * k + i]):.4f}')
        else:
            for i in range(ncols):
                for h in range(plot_dim):
                    ax[k, i].plot(x, fake_data[ncols * (k - 5) + i, :, h], color=color_list[h],
                                  label=params.dim_names[h])
                ax[k, i].plot(x, fake_score[ncols * (k - 5) + i], color=color_list[plot_dim],
                              label='disc_score')
                ax[k, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)
                ax[k, i].set_title(f'Generated data - score: {np.mean(fake_score[ncols * (k - 5) + i]):.4f}')

    f.savefig(os.path.join(plot_dir, f'disc_{plot_num}.png'))
    plt.close()


def model_list():
    """
    List all available models found under ./model.
    """
    files = os.listdir('./model')
    files = [name.replace('.py', '') for name in files if name.endswith('.py')]
    return files


def MinMaxScaler(data, min_val=None, max_val=None):
    """Min-Max Normalizer.
Args:
  - data: raw data
Returns:
  - norm_data: normalized data
  - min_val: minimum values (for renormalization)
  - max_val: maximum values (for renormalization)
"""
    if min_val is None:
        min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    if max_val is None:
        max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val
