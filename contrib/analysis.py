import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import torch


def resnet_cond(key, model):
    if 'weight' in key and key.replace('weight', 'running_mean') not in model.keys():
        #if 'module.conv1' not in key and 'module.fc' not in key:
        return True
    return False

def mobilenetv2_cond(key, model):
    if 'weight' in key and key.replace('weight', 'running_mean') not in model.keys():
        #if 'features.0.conv' not in key and 'classifier.0' not in key:
        return True
    return False


def plot_layer_weight(model, cond, result_path):
    fig, axes = plt.subplots(5, 12, sharex=True, sharey=True, figsize=(30, 10))
    n = 0
    for key in model.keys():
        if cond(key, model):
            mat = model[key].flatten().numpy()
            a = axes[n // 12, n % 12].hist(mat, bins=list(np.arange(-1,2,0.01)), density=1)

            max_prob = np.amax(np.absolute(a[0]))
            std = np.std(mat)
            axes[n // 12, n % 12].plot([std, std], [0, max_prob], color='green', linewidth=0.5)
            axes[n // 12, n % 12].plot([-std, -std], [0, max_prob], color='green', linewidth=0.5)
            axes[n // 12, n % 12].plot([np.max(mat), np.max(mat)], [0, max_prob], color='red', linewidth=0.5)
            axes[n // 12, n % 12].plot([np.min(mat), np.min(mat)], [0, max_prob], color='red', linewidth=0.5)
        
            #plt.ylabel('Probability')
            #plt.xlabel(key)
            n += 1
    plt.xlim([-1, 2])
    plt.tight_layout()
    #plt.savefig(str(result_path / (key + '.png')))
    plt.savefig(str(result_path / 'weights.png'))
    plt.close()

    # for key in model.keys():
    #     if cond(key, model):
    #         mat = model[key].flatten().numpy()
    #         a = plt.hist(mat, bins=201, density=1)

    #         max_prob = np.amax(np.absolute(a[0]))
    #         std = np.std(mat)
    #         plt.plot([std, std], [0, max_prob], color='green', linewidth=0.5)
    #         plt.plot([-std, -std], [0, max_prob], color='green', linewidth=0.5)
    #         plt.plot([np.max(mat), np.max(mat)], [0, max_prob], color='red', linewidth=0.5)
    #         plt.plot([np.min(mat), np.min(mat)], [0, max_prob], color='red', linewidth=0.5)
        
    #         plt.ylabel('Probability')
    #         plt.xlabel(key)
    
    #         plt.tight_layout()
    #         plt.savefig(str(result_path / (key + '.png')))
    #         plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--ckpt', type=str, default='best.pth',
                        help='checkpoint name')
    parser.add_argument('--path', type=str, default='logs/resnet56/cifar100/base/0',
                        help='target path')
    parser.add_argument('--model', type=str, default='resnet',
                        help='The name of model architecture \{resnet, mobilenetv2\}')
    args = parser.parse_args()

    # set paths
    model_path = Path(args.path)
    ckpt_path = model_path / args.ckpt
    result_path = model_path / 'figures'
    if not result_path.exists():
        result_path.mkdir()

    # load the trained model
    model = torch.load(ckpt_path, map_location='cpu')['model']
    if args.model == 'resnet':
        cond = resnet_cond
    elif args.model == 'mobilenetv2':
        cond = mobilenetv2_cond
    else:
        raise ValueError
    
    # layer-wise plot
    plot_layer_weight(model, cond, result_path)
