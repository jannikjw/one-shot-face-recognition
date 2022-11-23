# -*- coding: utf-8 -*-
""" main.py """

from experiments.example_config import CFG
from experiments.Experiment import Experiment


def run():
    """Builds model, loads data, trains and evaluates"""
    model = Experiment(CFG)
    model.load_data()
    train_inds, test_inds, vault_inds = model._create_train_test_split()
    model._load_image_train(train_inds)
    model._load_image_test(test_inds)
    model._load_image_vault(vault_inds)
    print(model.train_df.head())
    print(len(train_inds))
    print(model.test_df.head())
    print(len(test_inds))
    print(set(train_inds).intersection(test_inds))
    print(len(vault_inds))
    print(set(vault_inds).intersection(test_inds))
    print(len(set(vault_inds).intersection(train_inds)))

    #model.build()
    #model.train()
    #model.evaluate()


if __name__ == '__main__':
    run()