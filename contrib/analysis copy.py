import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--ckpt', type=str, default='400.best',
                        help='a checkpoint path')
    parser.add_argument('--path', type=str, default='400',
                        help='a result path')
    parser.add_argument('--bitw', type=int, default=4,
                        help='bits for weights')
    args = parser.parse_args()

    model = torch.load(args.ckpt, map_location='cpu')['model']
    bit = args.bitw
    q_n = -(2 ** (bit - 1))
    q_p = 2 ** (bit - 1) - 1

    result = Path(args.path)
    if not result.exists():
        result.mkdir()
    
    # layer-wise dictionary
    layer_dict = {}
    for key in model.keys():
        #if 'weight' in key and 'features.0.conv' not in key and 'classifier.0' not in key:
        if 'weight' in key and 'module.conv1' not in key and 'module.fc' not in key:
        
            step_key = key.replace('weight', 'quantizer.step')
            if step_key in model.keys():
                w = model[key]
                s = model[step_key]

                q = torch.clamp(w / s, q_n, q_p).round()
                qf = q.flatten()

                temp_dict = {}
                for i in range(q.nelement()):
                    n = int(qf[i])
                    if n in temp_dict.keys():
                        temp_dict[n] += 1
                    else:
                        temp_dict[n] = 1
                layer_dict[key] = temp_dict

    # total dictionary
    total_dict = {}
    for key in layer_dict.keys():
        for n in layer_dict[key].keys():
            if n in total_dict.keys():
                total_dict[n] += layer_dict[key][n]
            else:
                total_dict[n] = layer_dict[key][n]
    
    # layer plot
    for key in layer_dict.keys():
        al = np.sum([layer_dict[key][n] for n in layer_dict[key].keys()])
        tw = layer_dict[key][-1] + layer_dict[key][0] + layer_dict[key][1]

        prob_tw = tw / al * 100
        prob_slw = (al - tw) / al * 100

        txt = 'tw: {:.6f}%/ slw: {:.6f}%'.format(prob_tw, prob_slw)

        a = plt.hist(model[key].flatten().numpy(), bins=201, density=1)

        max_prob = np.amax(np.absolute(a[0]))
        sw = model[key.replace('weight', 'quantizer.step')].numpy()[0]

        plt.plot([q_n * sw, q_n * sw], [0, max_prob], color='red', linewidth=0.5)
        plt.plot([q_p * sw, q_p * sw], [0, max_prob], color='red', linewidth=0.5)
        plt.plot([-sw * 1.5, -sw * 1.5], [0, max_prob], color='green', linewidth=0.5)
        plt.plot([sw * 1.5, sw * 1.5], [0, max_prob], color='green', linewidth=0.5)

        plt.text(q_n * sw, max_prob, txt)
        plt.ylabel('Probability')
        plt.xlabel(key)
        plt.savefig(str(result / (key + '.png')))
        plt.close()

    # total plot
    al = np.sum([total_dict[n] for n in total_dict.keys()])
    tw = total_dict[-1] + total_dict[0] + total_dict[1]

    prob_tw = tw / al * 100
    prob_slw = (al - tw) / al * 100

    part1_keys = [key for key in total_dict.keys() if np.absolute(key) <= 1]
    part1_vals = [total_dict[key] / al for key in total_dict.keys() if np.absolute(key) <= 1]
    part2_keys = [key for key in total_dict.keys() if np.absolute(key) > 1]
    part2_vals = [total_dict[key] / al for key in total_dict.keys() if np.absolute(key) > 1]

    plt.figure(figsize=(3, 2.5))
    plt.bar(part1_keys, part1_vals, color='blue', alpha=0.5, width=0.6)
    plt.bar(part2_keys, part2_vals, color='red', alpha=0.5, width=0.6)

    #for key in total_dict.keys():
    #    plt.text(key, total_dict[key], str(total_dict[key]), horizontalalignment='center', fontsize=5)

    #txt = 'lv3: {:.3f}%/ etc: {:.3f}%'.format(prob_tw, prob_slw)
    x_temp = min([float(x) for x in part2_keys])
    y_temp = max([float(x) for x in part1_vals])
    #plt.text(x_temp, y_temp, txt)

    #plt.xticks([-8, -4, 0, 4, 7], ['-8', '-4', '0', '4', '7'], fontsize=10)
    plt.xticks([-2, 0, 1], ['-2', '0', '1'], fontsize=10)
    plt.yticks(fontsize=10)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.xlim(-9.0, 8.0)
    plt.ylabel('Probability', fontsize=12)
    plt.xlabel('Quantized weights', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(result / ('quantized.png')), bbox_inches='tight')
    plt.savefig(str(result / ('quantized.pdf')), bbox_inches='tight')
    plt.close()

    # layer & total txt
    total_al = np.sum([total_dict[n] for n in total_dict.keys()])
    total_tw = total_dict[-1] + total_dict[0] + total_dict[1]

    total_prob_tw = total_tw / total_al * 100
    total_prob_slw = (total_al - total_tw) / total_al * 100

    layer_prob_tw = []
    layer_nele = []
    for key in layer_dict.keys():
        al = np.sum([layer_dict[key][n] for n in layer_dict[key].keys()])
        tw = layer_dict[key][-1] + layer_dict[key][0] + layer_dict[key][1]

        prob_tw = tw / al * 100
        #prob_slw = (al - tw) / al * 100

        layer_nele.append(al)
        layer_prob_tw.append(prob_tw)

    with open(str(result / ('result.txt')), 'w') as file:
        file.write(str(total_dict))
        file.write('\n\n')
        file.write('Total TW Prob: {:.6f}\n'.format(total_prob_tw))
        file.write('Total SLW Prob: {:.6f}'.format(total_prob_slw))
        file.write('\n\n')

        file.write(str(list(layer_dict.keys())))
        file.write('\n')
        file.write(str(layer_nele))
        file.write('\n')
        file.write(str(layer_prob_tw))
        file.write('\n')
