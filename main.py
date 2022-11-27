# -*- coding: utf-8 -*-
""" main.py """

from experiments.example_config import CFG
from experiments.Experiment import Experiment


def run():
    """Define Model. Pass in config file CFG for particular Experiment
    model.train() - trains model, creates embeddings, and evaluates on test set (returns test accuracy)"""
    model = Experiment(CFG)
    model.train()

if __name__ == '__main__':
    run()