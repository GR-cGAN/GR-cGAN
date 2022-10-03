"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import logging
import math
import numpy as np
import os

import utils
from utils import extract_time, TS_predictor_sampler

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

logger = logging.getLogger('GAN.Pred_judge')


class Predictive_judge(Module):

    def __init__(self, input_dim, all_steps, predict_steps, device, d_model=20, nhead=4, num_blocks=2,
                 dropout=0.1, train_keep_ratio=1):
        super(Predictive_judge, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dim = input_dim
        dim_feedforward = int(d_model * 1.5)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_blocks,
                                          num_decoder_layers=num_blocks, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation='gelu', batch_first=True, device=device)
        self.pe = torch.zeros(all_steps, d_model, device=device)
        position = torch.arange(0, all_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.pe_given = self.pe[:, :-predict_steps]
        self.pe_pred = self.pe[:, -predict_steps:]
        self.pred_proj = nn.Linear(d_model, input_dim)
        self.predict_steps = predict_steps
        self.given_steps = all_steps - predict_steps
        self.d_model = d_model
        self.device = device
        self.train_keep_ratio = train_keep_ratio

    def forward(self, x, test=False):
        src = self.input_proj(x) + self.pe_given.detach()
        if not test:
            mask = torch.cuda.FloatTensor(x.shape[0], self.given_steps).uniform_() > self.train_keep_ratio
            src[mask] = -1
        tgt = torch.zeros(x.shape[0], self.predict_steps, self.d_model, device=self.device)
        out = self.transformer(src, tgt + self.pe_pred.detach())
        y_hat = self.pred_proj(out)
        return y_hat


def predictive_score_metrics(ori_data, generated_data, params, metric_itt):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    print('ori_data: ', ori_data.shape)
    print('gen_data: ', generated_data.shape)

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Build a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = params.pred_judge.hidden_dim
    predict_steps = params.pred_judge.predict_steps
    iterations = params.pred_judge.iterations
    batch_size = params.pred_judge.batch_size
    pred_batch_size = params.pred_judge.pred_batch_size
    num_layers = params.pred_judge.num_layers
    dropout = params.pred_judge.dropout
    learning_rate = params.pred_judge.learning_rate
    device = params.device
    test_itt = params.pred_judge.test_itt
    train_keep_ratio = params.pred_judge.train_keep_ratio

    predictor = Predictive_judge(dim, seq_len, predict_steps, device, d_model=hidden_dim, num_blocks=num_layers,
                                 dropout=dropout, train_keep_ratio=train_keep_ratio).to(device)

    # optimizer
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    mae_loss = nn.L1Loss()

    split_indices = np.random.choice([True, False], generated_data.shape[0], p=(0.85, 0.15))
    train_dataset = TS_predictor_sampler(generated_data[split_indices], predict_steps)
    # valid_dataset = TS_predictor_sampler(generated_data[~split_indices], predict_steps)
    train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=8)
    # valid_loader = DataLoader(valid_dataset, batch_size, sampler=SequentialSampler(train_dataset), num_workers=8,
    #                           drop_last=False)
    it = iter(train_loader)

    def eval_gan(eval_dataset):
        predictor.eval()
        with torch.no_grad():
            preds = torch.zeros(eval_dataset.shape[0], predict_steps, eval_dataset.shape[2], device=device)
            gt = eval_dataset[:, -predict_steps:]
            eval_pt_dataset = TS_predictor_sampler(eval_dataset, predict_steps)
            eval_loader = DataLoader(eval_pt_dataset, pred_batch_size, sampler=SequentialSampler(eval_pt_dataset),
                                     num_workers=8, drop_last=False)

            count = 0
            for i, batch in enumerate(tqdm(eval_loader)):
                pred_batch = predictor(batch[0].to(device), test=True)
                current_b_sz = pred_batch.shape[0]
                preds[count:count+current_b_sz] = pred_batch
                count += current_b_sz

            # Compute the performance in terms of MAE
            predictive_score = np.sum(np.abs(gt - preds.cpu().data.numpy())) / gt.size
        predictor.train()
        return predictive_score

    best_val_score = math.inf
    best_test_score = math.inf
    best_itt = -1

    # Training
    predictor.train()

    # Training using Synthetic dataset
    for itt in tqdm(range(iterations)):
        optimizer.zero_grad()

        # Set mini-batch
        try:  # https://stackoverflow.com/a/58876890/8365622
            # Samples the batch
            x_mb, y_mb = map(lambda x: x.to(device), next(it))
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            it = iter(train_loader)
            x_mb, y_mb = map(lambda x: x.to(device), next(it))

        # Train predictor
        y_hat_mb = predictor(x_mb)
        loss = mae_loss(y_mb, y_hat_mb)
        loss.backward()
        optimizer.step()

        if (itt + 1) % test_itt == 0:
            valid_score = eval_gan(generated_data[~split_indices])
            test_score = eval_gan(ori_data)
            if valid_score < best_val_score:
                best_itt = itt
                best_val_score = valid_score
                best_test_score = test_score
            print(f'Valid: {valid_score:.5f}, Test: {test_score:.5f}, best_test: {best_test_score:.5f}, '
                  f'best_itt = {best_itt}')

            if best_itt + 3000 < itt:
                print(f'Early stopping at iteration {itt}...')
                break

    # Test the trained model on the original data
    predictor.eval()
    with torch.no_grad():
        pred_y_curr = torch.zeros(ori_data.shape[0], predict_steps, ori_data.shape[2], device=device)
        y_mb = ori_data[:, -predict_steps:]
        eval_pt_dataset = TS_predictor_sampler(ori_data, predict_steps)
        eval_loader = DataLoader(eval_pt_dataset, batch_size, sampler=SequentialSampler(eval_pt_dataset),
                                 num_workers=8, drop_last=False)

        count = 0
        for i, batch in enumerate(tqdm(eval_loader)):
            pred_batch = predictor(batch[0].to(device), test=True)
            current_b_sz = pred_batch.shape[0]
            pred_y_curr[count:count+current_b_sz] = pred_batch
            count += current_b_sz

        # Prediction
        pred_y_curr = pred_y_curr.data.cpu().numpy()

        # Compute the performance in terms of MAE
        predictive_score = np.sum(np.abs(y_mb - pred_y_curr)) / y_mb.size

        # visualize discriminative score
        vis_select_index = np.random.choice(pred_y_curr.shape[0], size=8, replace=False)
        gt_plot_data = ori_data[vis_select_index]
        preds_plot_data = pred_y_curr[vis_select_index]
        utils.plot_predictive(os.path.join(params.model_dir, f'{params.judge}_figures'), params.judge,
                              metric_itt, gt_plot_data, preds_plot_data, params)

    return best_test_score
