import torch
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from args import parse_args
from hps import Hyperparameters
from model import SmartVocoder
import numpy as np
import librosa
import os
import time
import soundfile

torch.backends.cudnn.benchmark = False

data_path = "mels"
sample_path = "outputs"
load_path = "KOR_only_checkpoint.pth"
temp = 0.6
hop_size = 256

def build_model(hps):
    model = SmartVocoder(hps)
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def synthesize(model):
    for filename in os.listdir(data_path):
        mel = np.load(os.path.join(data_path, filename)).transpose(0,2,1)
        mel = (mel*100+20-100)/20
        mel = torch.tensor(mel).to(device)
        B, C, T = mel.shape
        z = torch.randn(1, 1, T*hop_size).to(device)
        z = z * temp
        torch.cuda.synchronize()
        timestemp = time.time()
        with torch.no_grad():
            y_gen = model.reverse(z, mel).squeeze()

        wav = y_gen.to(torch.device("cpu")).data.numpy()
        wav_name = '{}/{}.wav'.format(sample_path,  filename.split('.')[0])
        torch.cuda.synchronize()
        print('{} seconds'.format(time.time() - timestemp))
        soundfile.write(wav_name, wav, 22050)
        print('{} Saved!'.format(wav_name))


if __name__ == "__main__":
    global global_step
    global start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    hps = Hyperparameters(args)
    model = build_model(hps)
        
    print("Load checkpoint from: {}".format(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    print('sample_path', sample_path)
    with torch.no_grad():
        synthesize(model)
