"""
Code for "Mind Your Step: Continuous Conditional GANs with Generator Regularization".

Modified based on Time-series Generative Adversarial Networks (TimeGAN)
 TimeGAN is by Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

import argparse
import logging
import numpy as np
import os
import importlib
import warnings
import torch

from metrics_ts.discriminative_metrics_new import discriminative_score_metrics
from metrics_ts.predictive_metrics_new import predictive_score_metrics
from metrics_ts.visualization_metrics import visualization
import utils

warnings.filterwarnings("ignore")


def main(args):
    """Main function for timeGAN experiments.
  
      Args:
        - data_name: sine, stock, or energy
        - seq_len: sequence length
        - Network parameters (should be optimized for different datasets)
          - module: gru, lstm, or lstmLN
          - hidden_dim: hidden dimensions
          - num_layer: number of layers
          - iteration: number of training iterations
          - batch_size: the number of samples in each batch
        - metric_iteration: number of iterations for metric computation

      Returns:
        - ori_data: original data
        - generated_data: generated synthetic data
        - metric_results: discriminative and predictive scores
      """
    logger = logging.getLogger(f'GAN.{args.model}')
    net = importlib.import_module(f'model.{args.model}')
    data_loader = importlib.import_module(f'data.{args.data_name}.dataloader')

    # Load the parameters from json file
    data_dir = os.path.join('data', args.data_name)
    data_json_path = os.path.join(data_dir, 'params.json')
    assert os.path.isfile(data_json_path), f'No dataloader json configuration file found at {data_json_path}'
    params = utils.Params(data_json_path)

    if args.param_set is not None:
        model_dir = os.path.join('experiments', args.data_name, args.model, args.model_dir, args.param_set)
    else:
        model_dir = os.path.join('experiments', args.data_name, args.model, args.model_dir)
    params.base_model_dir = os.path.join('experiments', args.data_name, args.model, 'base_model')
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), f'No model json configuration file found at {json_path}'
    params.model_param = utils.Params(json_path)
    pred_judge_json_path = os.path.join('experiments', args.data_name,  args.judge, 'pred_params.json')
    disc_judge_json_path = os.path.join('experiments', args.data_name,  args.judge, 'disc_params.json')
    params.pred_judge = utils.Params(pred_judge_json_path)
    params.disc_judge = utils.Params(disc_judge_json_path)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info(f'args: {args}')

    params.data_name = args.data_name
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.judge = args.judge

    if args.param_set is not None:
        params.plot_title = os.path.join(args.data_name, args.model, args.model_dir, args.param_set)
    else:
        params.plot_title = os.path.join(args.data_name, args.model, args.model_dir)
    params.model = args.model

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(params.model_dir, f'{params.judge}_figures'))
    except FileExistsError:
        pass

    # Data loading
    train_data, ori_data = getattr(data_loader, 'data_loading')(params)
    print(args.data_name + ' dataset is ready.')

    # Synthetic data generation by TimeGAN
    # Set network parameters
    if torch.cuda.is_available():
        params.device = torch.device('cuda:0')
    else:
        params.device = torch.device('cpu')
    print(f'Training on device = {params.device}')

    model = net.Timegan(train_data, ori_data, params)
    model.fit(args.restore_iteration)
    print('Finish Synthetic Data Generation')

    # Performance metrics
    # Output initialization
    metric_results = dict()

    logger.info('===============================================================================')
    print(f'Begin {params.disc_judge.metric_iteration} iterations of discriminative score evaluation.')
    # 2. Discriminative Score
    discriminative_score = list()
    for i in range(params.disc_judge.metric_iteration):
        generated_data_disc = model.generate_data_given_size(model.disc_judge_multiplier)
        temp_disc = discriminative_score_metrics(ori_data, generated_data_disc, params, i)
        discriminative_score.append(temp_disc)
        logger.info(f'{args.model} discriminative score (iteration {i}): {temp_disc}')
        logger.info(f'All discriminative scores: {discriminative_score}')
    metric_results['discriminative'] = np.mean(discriminative_score)

    logger.info('===============================================================================')
    print(f'Begin {params.pred_judge.metric_iteration} iterations of predictive score evaluation.')
    # 3. Predictive score
    predictive_score = list()
    for i in range(params.pred_judge.metric_iteration):
        generated_data_pred = model.generate_data_given_size(model.pred_judge_multiplier)
        temp_pred = predictive_score_metrics(ori_data, generated_data_pred, params, i)
        predictive_score.append(temp_pred)
        logger.info(f'{args.model} predictive score (iteration {i}): {temp_pred}')
        logger.info(f'All predictive scores: {predictive_score}')
    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, generated_data_pred, 'pca')
    visualization(ori_data, generated_data_pred, 'tsne')

    # Print discriminative and predictive scores
    metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in metric_results.items())
    logger.info(f'- Full {args.model} metrics: {metrics_string}')

    return ori_data, generated_data_disc, metric_results


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name',
        choices=['stock', 'stock_cond', 'ETTm1', 'ETTm1_cond', 'ETTm1_cond_all', 'ETTm1_all'],
        default='ETTm1',
        type=str)
    parser.add_argument(
        '--model',
        choices=utils.model_list(),
        default='cTSGAN_gp',
        type=str)
    parser.add_argument(
        '--model-dir',
        choices=['base_model', 'param_search'],
        default='base_model',
        type=str)
    parser.add_argument(
        '--judge',
        choices=['TSGAN_judge'],
        default='TSGAN_judge',
        type=str)
    parser.add_argument(
        '--param-set',
        default=None,
        help='Set of model parameters created for hypersearch')
    parser.add_argument('--restore-iteration', default=-2, type=int,
                        help='Optional, which iteration to reload the weights from')

    args = parser.parse_args()

    # Calls main function
    ori_data, generated_data, metrics = main(args)
