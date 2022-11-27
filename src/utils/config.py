# -*- coding: utf-8 -*-
"""Config class"""

import json


class Config:
    """Config class which contains data, train, evaluate and model hyperparameters"""

    def __init__(self, data, train, evaluate, model):
        self.data = data
        self.train = train
        self.evaluate = evaluate
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.evaluate, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)