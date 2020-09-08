from text import *
import torch
import hyperparams as hparams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_melspec(target, melspec, melspec_post, mel_lengths):
    fig, axes = plt.subplots(3, 1, figsize=(20,30))
    T = mel_lengths[-1]

    axes[0].imshow(target[-1][:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1][:,:T],
                   origin='lower',
                   aspect='auto')

    axes[2].imshow(melspec_post[-1][:,:T],
                   origin='lower',
                   aspect='auto')

    return fig


def plot_alignments(alignments, mel_lengths, text_lengths, att_type, ref_mel_lengths=None):
    fig, axes = plt.subplots(hparams.n_layers, hparams.n_heads, figsize=(5*hparams.n_heads,5*hparams.n_layers))
    L, T = text_lengths[-1], mel_lengths[-1]
    if ref_mel_lengths==None:
        R = T
    else:
        R = ref_mel_lengths[-1]
    n_layers, n_heads = alignments.size(1), alignments.size(2)

    for layer in range(n_layers):
        for head in range(n_heads):
            if att_type=='enc':
                align = alignments[-1, layer, head].contiguous()
                axes[layer,head].imshow(align[:L, :L], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='dec':
                align = alignments[-1, layer, head].contiguous()
                axes[layer,head].imshow(align[:T, :T], aspect='auto')
                axes[layer,head].xaxis.tick_top()

            elif att_type=='enc_dec':
                align = alignments[-1, layer, head].transpose(0,1).contiguous()
                axes[layer,head].imshow(align[:L, :T], origin='lower', aspect='auto')

            elif att_type=='style':
                align = alignments[-1, layer, head].contiguous()
                axes[layer,head].imshow(align[:T, :R], aspect='auto')
                axes[layer,head].xaxis.tick_top()
       
    return fig


def plot_gate(gate_out):
    fig = plt.figure(figsize=(10,5))
    plt.plot(torch.sigmoid(gate_out[-1]))
    return fig
