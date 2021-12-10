import argparse
import sys
import yaml
from pathlib import Path


def add_quant_arguments(parser):
    # quantization configuration
    parser.add_argument('-Q', '--quant', '--quantization', dest='quantization', default='fp', type=str,
                        help='method of quantization to apply such as pact, lsq, etc (default: fp (full-precision))')
    parser.add_argument("--bitw", type=int, default=4, metavar="N",
                        help="Bit-width for weights")
    parser.add_argument("--bita", type=int, default=4, metavar="N",
                        help="Bit-width for activations")
    parser.add_argument("--first-conv-bitw", type=int, default=8, metavar="N",
                        help="Bit-width for weights of the first conv layer")
    parser.add_argument("--last-fc-bitw", type=int, default=8, metavar="N",
                        help="Bit-width for weights of the last fc layer")
    parser.add_argument('--symmetric', default=False, action='store_true',
                        help='symmetric or not for the quantization of weights and activations')

    parser.add_argument("--lsq-ewgs-fsh-k", type=float, default=100.,
                        help="Hyperparameter k for the lsq-ewgs-fsh method")

    # for pruning
    parser.add_argument('-P', '--prun', '--pruning', dest='pruning', default=None, type=str,
                        help='method for pruning to apply such as dpf (default: None)')
    parser.add_argument('--prun-type', default='weight', type=str, choices=['weight', 'filter', 'channel'],
                        help='specify the type of pruning, e.g., weight, filter, channel')
    parser.add_argument('--prun-freq', default=16, type=int,
                        help='mask update frequency during training')
    parser.add_argument('--ratio', default=0.5, type=float,
                         help='the percentage of weights to eliminate')
    parser.add_argument('--importance', default='L1', type=str, choices=['L1', 'L2', 'grad', 'syn'],
                         help='Importance Method : L1, L2, grad, syn')
    parser.add_argument("--first-conv", action='store_true', default=False,
                        help="prune the first convolution layer")
    parser.add_argument("--last-fc", action='store_true', default=False,
                        help="prune the last fully-connected layer")
