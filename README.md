# Codebase for "Mind Your Step: Continuous Conditional GANs with Generator Regularization"


This repo is based on the released code of "Time-series Generative Adversarial Networks (TimeGAN)".


This directory contains implementations of cTimeGAN framework for two real-world datasets.

-   ETTm1: https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTm1.csv
-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG

To run the pipeline for training and evaluation on cTimeGAN framwork, simply run 
python3 main_timegan.py. See main_timegan.py for a list of arguments that you can pass.

Note that in order to run cTSGAN_gp, you must train the vae network first. For example, to run
ETTm1, you must train vae on ETTm1_cond.

Note that any model architecture can be used as the generator and 
discriminator model such as RNNs or Transformers. 

### Command inputs:

-   data-name: ETTm1, ETTm1_all, stock, ETTm1_cond, ETTm1_all_cond, stock_cond
-   model-name: cTSGAN, cTSGAN_gp, vae, vae_gan

Note that network parameters should be optimized for different datasets.

### Example command
First, place ETTm1.csv at './data/raw_data/ETTm1/ETTm1.csv'.

For the ease of reimplementation, we included saved model weights in this repo. You can 
simply run
```shell
$ python3 main_timegan.py --data-name ETTm1 --model cTSGAN_gp  --restore-iteration 49000
```
to just train for another 1000 steps before continuing to evaluation.

If you want to train from scratch, first train vae on ETTm1_cond
```shell
$ python3 main_timegan.py --data-name ETTm1_cond --model vae_gan
```
Move the best saved model weights from experiments/ETTm1_cond/vae_gan/base_model/ to 
experiments/ETTm1/vae_gan/base_model/. Rename the file as 
saved_[cond_length]_klw_[klw].pth.tar. If you use the default parameters, this will be
saved_24_klw_00001.pth.tar.
```shell
$ python3 main_timegan.py --data-name ETTm1 --model cTSGAN_gp
```
