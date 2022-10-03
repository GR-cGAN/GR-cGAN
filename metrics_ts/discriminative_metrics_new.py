"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import logging
import numpy as np
import os

import utils
from utils import train_test_divide, extract_time, TS_sampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

logger = logging.getLogger('GAN.Disc_judge')


class Discriminative_judge(Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super(Discriminative_judge, self).__init__()
        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim, device=self.device)
        # c = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim, device=self.device)
        lstm_output, _ = self.lstm(x, h)
        y_hat_logit = self.linear(lstm_output)
        y_hat = F.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data, params, metric_itt):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data

    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Build a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = params.disc_judge.hidden_dim
    iterations = params.disc_judge.iterations
    batch_size = params.disc_judge.batch_size
    pred_batch_size = params.disc_judge.pred_batch_size
    num_layers = params.disc_judge.num_layers
    dropout = params.disc_judge.dropout
    learning_rate = params.disc_judge.learning_rate
    device = params.device
    test_itt = params.disc_judge.test_itt

    discriminator = Discriminative_judge(dim, hidden_dim, num_layers, dropout, device).to(device)

    # optimizer
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    bce_loss = nn.BCEWithLogitsLoss()

    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    split_indices = np.random.choice([True, False], test_t.shape[0], p=(0.5, 0.5))
    val_x = test_x[split_indices]
    val_x_hat = test_x_hat[split_indices]
    val_t = test_t[split_indices]
    val_t_hat = test_t_hat[split_indices]
    test_x = test_x[~split_indices]
    test_x_hat = test_x_hat[~split_indices]
    test_t = test_t[~split_indices]
    test_t_hat = test_t_hat[~split_indices]

    dataset = TS_sampler(train_x)
    sampler = RandomSampler(dataset)
    dataset_hat = TS_sampler(train_x_hat)
    sampler_hat = RandomSampler(dataset_hat)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=8)
    dataloader_hat = DataLoader(dataset_hat, batch_size, sampler=sampler_hat, num_workers=8)
    it = iter(dataloader)
    it_hat = iter(dataloader_hat)

    def eval_gan(eval_dataset, eval_dataset_hat):
        discriminator.eval()
        with torch.no_grad():
            _, y_pred_real_curr = discriminator(torch.from_numpy(np.float32(eval_dataset)).to(device))
            _, y_pred_fake_curr = discriminator(torch.from_numpy(np.float32(eval_dataset_hat)).to(device))

            y_pred_real_curr = y_pred_real_curr.data.cpu().numpy()
            y_pred_fake_curr = y_pred_fake_curr.data.cpu().numpy()

            y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
            y_label_final = np.concatenate((np.ones((y_pred_real_curr.shape[0], 1)),
                                            np.zeros((y_pred_fake_curr.shape[0], 1))), axis=0)

            # Compute the accuracy
            acc = np.sum(np.equal(y_label_final, (y_pred_final > 0.5))) / y_pred_final.size
            eval_score = np.abs(0.5 - acc)
        discriminator.train()
        return eval_score

    best_val_score = 0
    best_test_score = 0
    best_itt = -1

    # Training step
    discriminator.train()
    for itt in tqdm(range(iterations)):
        optimizer.zero_grad()

        # Set mini-batch
        try:  # https://stackoverflow.com/a/58876890/8365622
            # Samples the batch
            x_mb = next(it).to(device)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            it = iter(dataloader)
            x_mb = next(it).to(device)
        try:  # https://stackoverflow.com/a/58876890/8365622
            # Samples the batch
            x_hat_mb = next(it_hat).to(device)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            it_hat = iter(dataloader_hat)
            x_hat_mb = next(it_hat).to(device)

        # Train discriminator
        y_logit_real, y_pred_real = discriminator(x_mb)
        y_logit_fake, y_pred_fake = discriminator(x_hat_mb)

        # Loss for the discriminator
        d_loss_real = bce_loss(input=y_logit_real, target=torch.ones_like(y_logit_real))
        d_loss_fake = bce_loss(input=y_logit_fake, target=torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer.step()

        if (itt + 1) % test_itt == 0:
            valid_score = eval_gan(val_x, val_x_hat)
            test_score = eval_gan(test_x, test_x_hat)
            if valid_score > best_val_score:
                best_itt = itt
                best_val_score = valid_score
                best_test_score = test_score
            print(f'Valid: {valid_score:.5f}, Test: {test_score:.5f}, best_test: {best_test_score:.5f}, '
                  f'best_itt = {best_itt}')

            if best_itt + 1500 < itt:
                print(f'Early stopping at iteration {itt}...')
                break

    # Test the performance on the testing set
    discriminator.eval()
    with torch.no_grad():
        _, y_pred_real_curr = discriminator(torch.from_numpy(np.float32(test_x)).to(device))
        _, y_pred_fake_curr = discriminator(torch.from_numpy(np.float32(test_x_hat)).to(device))

        y_pred_real_curr = y_pred_real_curr.data.cpu().numpy()
        y_pred_fake_curr = y_pred_fake_curr.data.cpu().numpy()

        y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
        y_label_final = np.concatenate((np.ones((y_pred_real_curr.shape[0], 1)),
                                        np.zeros((y_pred_fake_curr.shape[0], 1))), axis=0)

        # Compute the accuracy
        acc = np.sum(np.equal(y_label_final, (y_pred_final > 0.5))) / y_pred_final.size
        discriminative_score = np.abs(0.5 - acc)

        # visualize discriminative score
        gen_select_index = np.random.choice(y_pred_fake_curr.shape[0], size=8, replace=False)
        real_select_index = np.random.choice(y_pred_real_curr.shape[0], size=8, replace=False)
        gen_plot_data = test_x_hat[gen_select_index, :, :]
        real_plot_data = test_x[real_select_index, :, :]
        gen_plot_score = y_pred_fake_curr[gen_select_index, :, :]
        real_plot_score = y_pred_real_curr[real_select_index, :, :]
        utils.plot_discriminative(os.path.join(params.model_dir, f'{params.judge}_figures'), params.judge,
                                  metric_itt, gen_plot_data, real_plot_data, gen_plot_score, real_plot_score, params)
    logger.info(f'Discriminative judge best test score: {best_test_score}')
    return best_test_score
