"""
Copyright (c) 2021-present Jinbae Park, Machine Learning and Visual Computing (MLVC), Kyung Hee University.
MIT license
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_regularizer(cfg):
    if cfg.regularizer == None:
        return None
    elif cfg.regularizer == 'l2':
        return L2_Regularizer(cfg)
    elif cfg.regularizer == 'l1':
        return L1_Regularizer(cfg)
    elif cfg.regularizer == 'elasticnet':
        return ElasticNet_Regularizer(cfg)
    elif cfg.regularizer == 'smoothl1':
        return SmoothL1_Regularizer(cfg)
    else:
        raise ValueError


class Regularizer(object):
    def __init__(self, cfg):
        self.layer_types = [nn.Conv2d, nn.Linear]
        self.lmbd = cfg.lmbd

    def calculate(self, weight):
        pass

    def loss(self, model):
        losses = []
        for n, m in model.named_modules():
            if type(m) in self.layer_types:
                losses.append(self.calculate(m.weight))
        return self.lmbd * torch.sum(torch.stack(losses))


class L2_Regularizer(Regularizer):
    def __init__(self, cfg, **kwargs):
        super(L2_Regularizer, self).__init__(cfg)

    def calculate(self, weight):
        return 0.5 * weight.square().sum()


class L1_Regularizer(Regularizer):
    def __init__(self, cfg, **kwargs):
        super(L1_Regularizer, self).__init__(cfg)

    def calculate(self, weight):
        return weight.abs().sum()


class ElasticNet_Regularizer(Regularizer):
    # Elastic-Net regularization
    def __init__(self, cfg, **kwargs):
        super(ElasticNet_Regularizer, self).__init__(cfg)
        self.l1_lmbd = 1e-6 / cfg.lmbd

    def calculate(self, weight):
        return 0.5 * weight.square().sum() + self.l1_lmbd * weight.abs().sum()


class SmoothL1_Regularizer(Regularizer):
    # Smooth L1 regularization
    def __init__(self, cfg, **kwargs):
        super(SmoothL1_Regularizer, self).__init__(cfg)
        self.beta = kwargs['beta'] if 'beta' in kwargs.keys() else 1

    def calculate(self, weight):
        loss = torch.where(weight.abs() < self.beta,
                           0.5 * weight.square() / self.beta,
                           weight.abs() - 0.5 * self.beta)
        return loss.sum()