import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_weight_threshold(model, rate, args):
    importance_all = None
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'mask' not in name:
            weights = item.data.view(-1).cpu()
            grads = item.grad.data.view(-1).cpu()

            if args.prune_imp == 'L1':
                importance = weights.abs().numpy()
            elif args.prune_imp == 'L2':
                importance = weights.pow(2).numpy()
            elif args.prune_imp == 'grad':
                importance = grads.abs().numpy()
            elif args.prune_imp == 'syn':
                importance = (weights * grads).abs().numpy()
            

            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
    return threshold


def weight_prune(model, threshold, args):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'weight' in name:
            key = name.replace('weight', 'mask')
            if key in state.keys():
                if args.prune_imp == 'L1':
                    mat = item.data.abs()
                elif args.prune_imp == 'L2':
                    mat = item.data.pow(2)
                elif args.prune_imp == 'grad':
                    mat = item.grad.data.abs()
                elif args.prune_imp == 'syn':
                    mat = (item.data * item.grad.data).abs()
                state[key].data.copy_(torch.gt(mat, threshold).float())


def get_filter_mask(model, rate, args):
    importance_all = None
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'weight' in name:
            filters = item.data.view(item.size(0), -1).cpu()
            weight_len = filters.size(1)
            if args.prune_imp =='L1':
                importance = filters.abs().sum(dim=1).numpy() / weight_len
            elif args.prune_imp == 'L2':
                importance = filters.pow(2).sum(dim=1).numpy() / weight_len
        
            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance)[int(len(importance) * rate)]
    #threshold = np.percentile(importance, rate)
    filter_mask = np.greater(importance, threshold)
    return filter_mask


def filter_prune(model, filter_mask):
    idx = 0
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'mask' in name:
            for i in range(item.size(0)):
                item.data[i,:,:,:] = 1 if filter_mask[idx] else 0
                idx+=1


def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if 'weight' in name or 'bias' in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity



def regularize_th(epoch, model, threshold, prune_rate, reg_lmbd):
    #curr_prune_rate = cal_curr_prune_rate(epoch, prune_rate)
    curr_prune_rate = prune_rate
    upper_bound = cal_upper_bound(model, curr_prune_rate)
    
    losses = []
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'mask' in name:
            key = name.replace('mask', 'weight')
            losses.append(cal_pl1_th(state[key], threshold, upper_bound))
    loss = torch.stack(losses).sum()
    
    #curr_reg_lmbd = cal_curr_lmbd2(epoch, reg_lmbd)
    curr_reg_lmbd = reg_lmbd
    return reg_lmbd * loss


def cal_curr_prune_rate(epoch, prune_rate):
    if epoch < 5:
        return prune_rate * (epoch / 5)
    else:
        return prune_rate
    

def cal_curr_lmbd2(epoch, reg_lmbd):
    if epoch < 5:
        return reg_lmbd * (1 - (1 - epoch / 5) ** 0.5)
    else:
        return reg_lmbd


def cal_upper_bound(model, prune_rate):
    with torch.no_grad():
        all = []
        state = model.state_dict()
        for name, item in model.named_parameters():
            if 'mask' in name:
                key = name.replace('mask', 'weight')
                weight = state[key].data.view(-1)
                importance = weight.abs()
                all.append(importance)

        all = torch.cat(all)
        sorted, idx = all.sort(descending=True)
        if prune_rate == 0:
            upper_bound = sorted[-1]
        else:
            upper_bound = sorted[int(all.size(0) * (1 - prune_rate))]
        #print(sorted[0], '\t', sorted[-1], '\t', upper_bound)
    return upper_bound


def cal_pl1_th(weight, threshold, upper_bound):
    weight_abs = weight.abs()
    #loss = torch.where((weight_abs > threshold) & (weight_abs < upper_bound), weight_abs, torch.zeros_like(weight)).sum()
    loss = torch.where(weight_abs > threshold, 0.5 * weight.pow(2), torch.zeros_like(weight)).sum()
    return loss
