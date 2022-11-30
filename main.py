# -*- coding: utf-8 -*-
""" main.py """

from experiments.config_GAN import CFG_GAN
from experiments.config_no_GAN import CFG

from experiments.Experiment import Experiment


def run(is_gan):
    """Define Model. Pass in config file CFG for particular Experiment
    model.train() - trains model, creates embeddings, and evaluates on test set (returns test accuracy)"""

    if is_gan=='True':
        model = Experiment(CFG_GAN)
        model.train()
    else:
        model = Experiment(CFG)
        model.train()

if __name__ == '__main__':
    import sys

    is_gan = sys.argv[1]
    
    run(is_gan)