"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
timegan.py
Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, RandomSampler
from torch import Tensor
from typing import List, Tuple

from tqdm import tqdm

import utils

matplotlib.use('Agg')
# matplotlib.rcParams['savefig.dpi'] = 300  # Uncomment for higher plot resolutions
logger = logging.getLogger('GAN.VAE')


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 seq_len: int,
                 hidden_dim: int,
                 device,
                 total_itt: int,
                 max_kl_weight: float = 1):
        super(VanillaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device
        self.current_itt = 0
        self.warmup_kl_itt = total_itt // 10
        self.kl_schedule = self.frange_cycle_cosine(0.0, 1, total_itt, 10) * max_kl_weight
        self.kl_schedule[:self.warmup_kl_itt] = 0

        modules = []
        hidden_dims = [in_channels, 2 * hidden_dim, 4 * hidden_dim, 8 * hidden_dim]
        self.last_hidden_dim = hidden_dims[-1]

        # Build Encoder
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1],
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.latent_multiplier = math.ceil(self.seq_len / (2 ** 3))
        self.fc_mu = nn.Linear(self.last_hidden_dim * self.latent_multiplier, latent_dim)
        self.fc_var = nn.Linear(self.last_hidden_dim * self.latent_multiplier, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.last_hidden_dim * self.latent_multiplier)

        hidden_dims[0] = self.in_channels * 5
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*(modules + [nn.Dropout(0.1)]))
        self.final_layer1 = nn.Linear(5 * self.in_channels * self.seq_len, self.in_channels * self.seq_len)
        self.final_layer2 = nn.Linear(self.in_channels * self.seq_len, self.in_channels * self.seq_len)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # input:  torch.Size([128, 24, 28])
        result = self.encoder(input.permute(0, 2, 1))
        # result: torch.Size([128, 192, 3])
        result = torch.flatten(result, start_dim=1)
        # flatten:  torch.Size([128, 576])

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # decoder_input:  torch.Size([128, 576])
        result = result.view(-1, self.last_hidden_dim, self.latent_multiplier)
        result = self.decoder(result)
        result = torch.flatten(result, start_dim=1)
        # print('result: ', result.shape)
        result = self.final_layer2(self.final_layer1(result)).reshape(-1, self.seq_len, self.in_channels)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def frange_cycle_cosine(self, start, stop, total_itt, n_cycle=4, ratio=0.5):
        L = np.ones(total_itt)
        period = total_itt / n_cycle
        step = (stop - start) / (period * ratio)  # step is in [0,1]

        # transform into [0, pi] for plots:

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 0.5 - .5 * math.cos(v * math.pi)
                v += step
                i += 1
        return L

    def loss_fn(self, recons, gt, mu, log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :return:
        Args:
            log_var:
            mu:
        """
        recons_loss = F.mse_loss(recons, gt)

        if self.current_itt > self.warmup_kl_itt:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        else:
            kld_loss = torch.zeros(1, device=self.device)
        loss = recons_loss + self.kl_schedule[self.current_itt] * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, latent_z) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        samples = self.decode(latent_z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


def timegan(ori_data, params, restore_iteration=-2):
    """TimeGAN function.
    Use original data as training set to generater synthetic data (time-series)
    Args:
        - ori_data: original time-series data
        - parameters: TimeGAN network parameters
    Returns:
        - generated_data: generated time-series data
    """

    # Basic Parameters
    no, seq_len, input_dim = ori_data.shape
    assert input_dim == params.input_dim, f'input_dim not match in data_params.'

    # Network Parameters
    hidden_dim = params.model_param.hidden_dim  # input_dim * 4
    iterations = params.model_param.iterations
    batch_size = params.model_param.batch_size
    max_kl_weight = params.model_param.max_kl_weight
    z_dim = params.model_param.z_dim  # input_dim // 2
    learning_rate = params.model_param.learning_rate
    disc_judge_multiplier = params.disc_judge_multiplier
    pred_judge_multiplier = params.pred_judge_multiplier
    device = params.device

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = utils.extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer.
    Args:
      - data: raw data
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
        min_val = np.min(np.min(data, axis=0), axis=0)
        min_val = 0
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        max_val = 1
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Define networks
    vae = VanillaVAE(input_dim, z_dim, max_seq_len, hidden_dim, device=device, total_itt=iterations,
                     max_kl_weight=max_kl_weight).to(device)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    # to visualize loss in the third stage
    loss_summary = np.zeros((iterations, params.model_param.num_loss))
    loss_cache = np.zeros((params.model_param.save_iteration, params.model_param.num_loss))
    if restore_iteration != -2:
        restore_iteration = restore_iteration // params.model_param.save_iteration * params.model_param.save_iteration
        restore_path = os.path.join(params.model_dir, f'itt_{restore_iteration}_vae.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, vae, vae_optim)
        loss_summary[: restore_iteration + 1] = np.load(restore_path)

    start_iteration = restore_iteration + 2
    dataset = utils.TS_sampler(ori_data)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=8)
    it = iter(dataloader)
    logger.info('Start VAE Training')
    for itt in tqdm(range(start_iteration, iterations)):
        vae.current_itt = itt
        itt_cache = itt % params.model_param.save_iteration

        vae_optim.zero_grad()

        # Set mini-batch
        try:  # https://stackoverflow.com/a/58876890/8365622
            # Samples the batch
            x_mb = next(it).to(device)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            it = iter(dataloader)
            x_mb = next(it).to(device)
        recons, mu, log_var = vae(x_mb)
        loss = vae.loss_fn(recons, x_mb.detach(), mu, log_var)
        loss['loss'].backward()
        loss_cache[itt_cache, 0] = loss['loss'].item()
        loss_cache[itt_cache, 1] = loss['Reconstruction_Loss'].item()
        loss_cache[itt_cache, 2] = loss['KLD'].item()
        vae_optim.step()

        # Checkpoint
        if (itt + 1) % 500 == 0:
            logger.info(f'step: {itt}/{iterations}, loss: {loss["loss"].item():.4f},'
                        f' recons_loss: {loss["Reconstruction_Loss"].item():.4f},'
                        f' KLD: {loss["KLD"].item():.4f}')

        if (itt + 1) % params.model_param.save_iteration == 0:
            loss_summary[itt + 1 - params.model_param.save_iteration: itt + 1] = loss_cache.copy()
            np.save(os.path.join(params.model_dir, f'itt_{itt}_loss'), loss_summary[: itt + 1])
            plot_all_loss(loss_summary[: itt + 1], itt, params, location=params.plot_dir)

            utils.save_checkpoint({'epoch': itt,
                                   'state_dict': vae.state_dict(),
                                   'optim_dict': vae_optim.state_dict()},
                                  itt=itt,
                                  checkpoint=params.model_dir,
                                  ins_name='vae')

            with torch.no_grad():
                vae.eval()
                sampled_x = ori_data[np.random.choice(ori_data.shape[0], size=4, replace=False), :, :]
                x_mb = torch.from_numpy(np.float32(sampled_x)).to(device)
                recons, mu, log_var = vae(x_mb)

                z = torch.randn(12, z_dim, device=device)
                generated_data = vae.sample(z).data.cpu().numpy()
                generated_data = generated_data * max_val
                generated_data = generated_data + min_val
                vae.train()

            utils.plot_samples_vae(params.plot_dir, itt, sampled_x, recons.data.cpu().numpy(), mu.data.cpu().numpy(),
                                   log_var.data.cpu().numpy(), z.data.cpu().numpy(), generated_data, params)

    logger.info('Finish VAE Training')

    # Synthetic data generation
    vae.eval()

    def generate_data_given_size(multiplier, original_data_np):
        with torch.no_grad():
            gen_size = no * multiplier
            generated_data = torch.zeros(gen_size, seq_len, input_dim, device=device)
            original_data = torch.from_numpy(np.float32(np.tile(original_data_np, (multiplier, 1, 1)))).to(device)
            predict_batch_size = 512
            for i in range(predict_batch_size, gen_size, predict_batch_size):
                z = torch.randn(predict_batch_size, z_dim, device=device)
                generated_data[i - predict_batch_size: i] = vae.sample(z)

            left_batch_size = gen_size % predict_batch_size
            if left_batch_size > 0:
                z = torch.randn(left_batch_size, z_dim, device=device)
                generated_data[-left_batch_size:] = vae.sample(z)

            # Renormalization
            generated_data = generated_data.data.cpu().numpy() * max_val
            generated_data = generated_data + min_val
        return generated_data

    gen_data_disc = generate_data_given_size(disc_judge_multiplier, ori_data)
    gen_data_pred = generate_data_given_size(pred_judge_multiplier, ori_data)

    return gen_data_disc, gen_data_pred


def cum_by_axis1(input_x):
    cum_input = np.zeros(input_x.shape)
    for i in range(cum_input.shape[1]):
        cum_input[:, i] = np.sum(input_x[:, :(i + 1)], axis=1)
    return cum_input


def plot_all_loss(loss_summary, plot_num, params, location='./figures/'):
    gaussian_window_size = 1
    num_itt = loss_summary.shape[0]
    color_list = ['b', 'g', 'r', 'c', 'darkred', 'silver', 'm', 'y', 'b', 'pink']
    x = np.arange(num_itt)
    f = plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = f.subplots(2)
    f.suptitle(f'{params.plot_title}/it_{plot_num}')

    # total loss
    total_loss = gaussian_filter1d(loss_summary[:, 0], gaussian_window_size, axis=0)
    ax[1].plot(x, total_loss, color=color_list[0], label='total_loss')
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)

    # GS loss
    num_loss = 2
    loss_comp = gaussian_filter1d(loss_summary[:, 1:], gaussian_window_size, axis=0)
    loss_list = ['reconstruction loss', 'KLD']
    for i in range(num_loss):
        ax[0].plot(x, loss_comp[:, i], color=color_list[i + 1], label=loss_list[i])
    ax[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True)

    f.savefig(os.path.join(location, f'loss_summary.png'))
    plt.close()
