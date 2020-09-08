import hyperparams as hparams
from torch.utils.data import DataLoader
#from .data_utils import TextMelSet, TextMelCollate
import torch
from text import *
import matplotlib.pyplot as plt
import os
import numpy as np
'''
def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelSet(hparams.training_files, hparams)
    valset = TextMelSet(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset,
                              num_workers=hparams.n_gpus-1,
                              shuffle=True,
                              batch_size=hparams.batch_size, 
                              drop_last=True, 
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(valset,
                            batch_size=hparams.batch_size//hparams.n_gpus,
                            collate_fn=collate_fn)
    
    return train_loader, val_loader, collate_fn


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
#    model_for_loading = checkpoint_dict['model']
#    model.load_state_dict(model_for_loading.state_dict())
    model.load_state_dict(checkpoint_dict['state_dict'])
    print("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    return model, optimizer, iteration

    
def lr_scheduling(opt, step, init_lr=hparams.lr, warmup_steps=hparams.warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * min(step ** -0.5, step * warmup_steps ** -1.5)
    return


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len))
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    return mask
'''

def reorder_batch(x, n_gpus):
    assert (x.size(0)%n_gpus)==0, 'Batch size must be a multiple of the number of GPUs.'
    new_x = x.new_zeros(x.size())
    chunk_size = x.size(0)//n_gpus
    
    for i in range(n_gpus):
        new_x[i::n_gpus] = x[i*chunk_size:(i+1)*chunk_size]
    
    return new_x
'''
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
#    Sinusoid position encoding table

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
'''
