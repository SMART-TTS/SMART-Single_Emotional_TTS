import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text.HangulUtilsHrim import hangul_to_sequence
import collections
from scipy import signal
import torch as t
import math
import argparse
from g2pk import G2p as g2p

class KORDatasets(Dataset):
    """KOR_DB dataset"""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.g2p = g2p()

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        preprocess_name = os.path.join(hp.preprocess_path, self.landmarks_frame.iloc[idx, 0]) # ##i preprocessed/00001
        text = self.landmarks_frame.iloc[idx, 1]
        fname = self.landmarks_frame.iloc[idx, 0]
        text = np.asarray(hangul_to_sequence(text, self.g2p), dtype=np.int32)
        mel = np.load(preprocess_name + '.pt.npy')
        mag = np.load(preprocess_name + '.mag.npy')

        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)
        mel_length = len(mel)
        sample = {'text': text, 'mel': mel, 'mag' : mag, 'text_length':text_length, 'mel_length':mel_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text, 'fname':fname}

        return sample

class PostKORDatasets(Dataset):
    """KORSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
#        preprocess_name = os.path.join(hp.preprocess_path, "{:05d}".format(self.landmarks_frame.iloc[idx, 0]))
        preprocess_name = os.path.join(hp.preprocess_path, self.landmarks_frame.iloc[idx, 0])
        mel = np.load(preprocess_name + '.pt.npy')
        mag = np.load(preprocess_name + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text= [d['pos_text'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        fname = [d['fname'] for d in batch]
       
        
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mag = [i for i, _ in sorted(zip(mag, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_length = [i for i, _ in sorted(zip(mel_length, text_length), key=lambda x: x[1], reverse=True)]
        fname = [i for i, _ in sorted(zip(fname, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_text = _prepare_data(pos_text).astype(np.int32)


        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mag), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length), t.LongTensor(mel_length), fname

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

DB = "KOR" # ##i "KOR" or "LJ"
def get_dataset(data_csv):
    if DB == "KOR":
        return KORDatasets(os.path.join(hp.data_path, data_csv), os.path.join(hp.data_path, 'wav_org'))
    else:
        return LJDatasets(os.path.join(hp.data_path, 'metadata_jka.csv'), os.path.join(hp.data_path, 'wavs'))

def get_stoptoken_dataset(data_csv):
    if DB == "KOR":
        return StopKORDatasets(os.path.join(hp.data_path, data_csv), os.path.join(hp.data_path, 'wav_org'))
    else:
        return LJDatasets(os.path.join(hp.data_path, 'metadata_jka.csv'), os.path.join(hp.data_path, 'wavs'))

def get_post_dataset(data_csv):
    if DB == "KOR":
        return PostKORDatasets(os.path.join(hp.data_path, data_csv), os.path.join(hp.data_path,'wav_org'))
    else:
        return PostLJDatasets(os.path.join(hp.data_path, 'metadata_jka.csv'), os.path.join(hp.data_path, 'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

