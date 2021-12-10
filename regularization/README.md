# How to use regularization module

1. add configuration
```
parser.add_argument('--rg', '--regularizer', dest='regularizer', default=None, type=str,
                    help='regularizer for weight decay (default: None) \{l2, l1\}')
parser.add_argument('--lmbd', default=1e-4, type=float,
                    help='the hyperparameter for the regularizer (default: 1e-4)')
```

2. set regularizer and calculate loss
```
from regularization import set_regularizer
...
regularizer = set_regularizer(cfg)
...
if regularizer != None:
    loss += regularizer.loss(model)
```

3. run a training
```
# Example commands
# L2 regularization
#python main.py --name l2_1e-0 --idx 0 -g 0 -j 8 --dataset cifar100 --datapath ../data --arch resnet --layers 56 --batch-size 128 --run-type train --epochs 300 --lr 0.1 --sched cosine --sched-batch --regularizer l2 --lmbd 1

# Elasticnet
#python main.py --name elasticnet_1e-2 --idx 1 -g 0 -j 8 --dataset cifar100 --datapath ../data --arch resnet --layers 56 --batch-size 128 --run-type train --epochs 300 --lr 0.1 --sched cosine --sched-batch --regularizer elasticnet --lmbd 1e-2
```