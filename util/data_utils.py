import random
import numpy as np
import hyperparams as hparams
import torch
import torch.utils.data
import torch.nn.functional as F
import os
import pickle as pkl

#from text import text_to_sequence


def load_filepaths_and_text(metadata, split="|"):
    with open(metadata, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


class TextMelSet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self.data_type=hparams.data_type

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        file_name = audiopath_and_text[0]
        seq = os.path.join(hparams.data_path, self.data_type)
        mel = os.path.join(hparams.data_path, 'melspectrogram')

        with open(f'{seq}/{file_name}_sequence.pkl', 'rb') as f:
            text = pkl.load(f)
        with open(f'{mel}/{file_name}_melspectrogram.pkl', 'rb') as f:
            mel = pkl.load(f)
        
        return (text, mel)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include Spec padded and gate padded
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
        gate_padded = torch.zeros(len(batch), max_target_len)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        
        return text_padded, input_lengths, mel_padded, output_lengths, gate_padded
