from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa
from multiprocessing import cpu_count
import argparse
import re
import random
import glob

random.seed(1234)

def build_from_path(in_dir, out_dir, hop_length, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    for folder in glob.iglob(in_dir + '/**/wav_22', recursive=True):
        for wav_name in os.listdir(folder):
            wav_path = os.path.join(folder, wav_name)
            save_dir = folder.replace(in_dir+'/',"")
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, save_dir, index, wav_path, hop_length)))
            index += 1
    return [future.result() for future in futures]

def _process_utterance(out_dir, save_dir, index, wav_path, hop_length):
    if hop_length == 256:
        sr = 22050
        fft_size = 2048
        win_length = 1024
        window='hann'
        fmin = 50
        fmax = 11000
        n_mels = 80
        eps = 1e-10
    else:
        import sys
        sys.exit('hop_length should be 256 !')

    wav, _ = librosa.load(wav_path, sr=sr)
    wav, _ = librosa.effects.trim(wav, top_db=35)
    out = wav / np.abs(wav).max() * 0.999
    linear = librosa.stft(y=out,
                        n_fft=fft_size,
                        window=window,
                        hop_length=hop_length,
                        win_length=win_length,)
    mag = np.abs(linear)
    mel_basis = librosa.filters.mel(sr, fft_size, n_mels, fmin, fmax)
    mel = np.dot(mel_basis, mag).T  # (t, n_mels)
    mel_spectrogram = np.log10(np.maximum(eps, mel))

    pad = (out.shape[0] // hop_length + 1) * hop_length - out.shape[0]
    pad_l = pad // 2
    pad_r = pad // 2 + pad % 2

    # zero pad for quantized signal
    out = np.pad(out, (pad_l, pad_r), mode="constant", constant_values=0)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * hop_length, "len(out) >= N  * hop_length"

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * hop_length]
    assert len(out) % hop_length == 0, "len(out) `%` hop_length == 0"

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_path= os.path.join(save_dir, 'audio-%05d.npy' % index)
    mel_path = os.path.join(save_dir, 'mel-%05d.npy' % index)

    if not os.path.isdir(os.path.join(out_dir, save_dir)):
        os.makedirs(os.path.join(out_dir, save_dir), exist_ok=True)

    np.save(os.path.join(out_dir, audio_path),
            out.astype(np.float32), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_path),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_path, mel_path, timesteps


def preprocess(in_dir, out_dir, hop_length, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, hop_length, num_workers)
    write_metadata(metadata, out_dir, hop_length)


def write_metadata(metadata, out_dir, hop_length):
    if hop_length == 256:
        sr = 22050
    random.shuffle(metadata)
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])        
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', '-i', type=str, default='DB', help='In Directory')
    parser.add_argument('--hop_length', '-hl', type=int, default=256, help='Hop size')
    parser.add_argument('--out_dir', '-o', type=str, default='datasets/preprocessed', help='Out Directory')
    args = parser.parse_args()


    num_workers = cpu_count() - 5
    print('<--- Preprocess start --->')
    preprocess(args.in_dir, args.out_dir, args.hop_length, num_workers)
    print('<--- Preprocess done --->')
