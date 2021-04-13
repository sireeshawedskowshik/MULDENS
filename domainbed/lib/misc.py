# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile

import numpy as np
import torch
import tqdm
from collections import Counter

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist() 
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            try:
                p = network.predict(x)
            except:
                p = network(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total
def invenio_accuracy(algorithm,eval_dict, test_envs,correct_models_selected_for_each_domain,device):
    correct = 0
    total = 0
    weights_offset = 0 
    eval_loader_names= list(eval_dict.keys())
    

    test_env = test_envs[0]

    obs_loader_insplit_names = ['env{}_in'.format(i)
        for i in range(len(eval_loader_names)//2) if i not in test_envs]
    obs_loader_outsplit_names= ['env{}_out'.format(i)
        for i in range(len(eval_loader_names)//2) if i not in test_envs]

    un_obs_insplit_name = ['env{}_in'.format(i) for i in test_envs]
    un_obs_outsplit_name = ['env{}_out'.format(i) for i in test_envs]

    
    for network_i in algorithm.invenio_networks:
        network_i.eval()

    # for observed domains, we know what models to select. So directly get the accuracies from corresponding models
    results={}
    for i in range(len(eval_loader_names)//2):
        if i not in test_envs:
            for split in ['_in','_out']:
                name = 'env'+str(i)+split
                loader= eval_dict[name][0]
                weights= eval_dict[name][1]
                model_num = int(correct_models_selected_for_each_domain[i])
                acc=accuracy(algorithm.invenio_networks[model_num],loader,weights,device)
                results[name+'_acc'] = acc


    return results
    # domains_selected_for_each_model=  [[] for i in range(len(algorithm.invenio_networks))]
    # for m in range(len(algorithm.invenio_networks)):
    #     for i,ms in enumerate(correct_models_selected_for_each_domain):
    #         if ms is not np.nan:
    #             if ms ==m:
    #                 domains_selected_for_each_model[m].append(i)
    # # compute betas
    # for i, model in enumerate(algorithm.invenio_networks):
    #     # compute gradients with the corresponding domains selected for this model
    #     domains_selected = domains_selected_for_each_model[i]
    #     obs_loaders= []
    #     for d in domains_selected: # train_loader
    #         loader_name= 'env'+str(d)+'_in'
    #         obs_loaders.append(eval_dict[loader_name])



    # with torch.no_grad():

        






    #     for x, y in loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         p = network.predict(x)
    #         if weights is None:
    #             batch_weights = torch.ones(len(x))
    #         else:
    #             batch_weights = weights[weights_offset : weights_offset + len(x)]
    #             weights_offset += len(x)
    #         batch_weights = batch_weights.to(device)
    #         if p.size(1) == 1:
    #             correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
    #         else:
    #             correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
    #         total += batch_weights.sum().item()
    # network.train()

    # return correct / total
class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')