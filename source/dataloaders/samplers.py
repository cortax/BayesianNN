import torch
import numpy as np
import torch.utils.data as Data

def getWeightedRandomSampler(list_targets):
    class_sample_count = np.unique(list_targets, return_counts=True)[1]
    inv_weight = 1. / class_sample_count
    #print(inv_weight)
    samples_weight = inv_weight[list_targets]
    #print('samples_weight:', samples_weight)
    samples_weight = torch.from_numpy(samples_weight)
    #WeightedRandomSampler returns a warning about tensor cloning...
    sampler = Data.WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    return sampler