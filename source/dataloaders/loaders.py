import torch
import numpy as np
import torch.utils.data as tData


def get_generic_loader(dataset, batch_size, 
                        random_seed=42, shuffle=False, sampler=None,
                        num_workers=4, pin_memory=False):
    torch.manual_seed(random_seed)
    dataloader = tData.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_train_valid_test_generic_loaders(train_dataset, valid_dataset, test_dataset,
                                            batch_size, shuffle_train=True, train_sampler=None,
                                            random_seed=42, num_workers=4, pin_memory=False):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Making all 3 Loaders with Training Shuffle at: {}'.format(shuffle_train))
    train_loader = tData.DataLoader(train_dataset, 
        batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = tData.DataLoader(valid_dataset, 
        batch_size=batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

    test_loader = tData.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader