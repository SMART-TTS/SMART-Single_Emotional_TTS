import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
import hyperparams as hp
import librosa

# ##i hrim fixes : mel.pt, mag.mag are saved in {projectDir}/preprocessed

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        fname = self.landmarks_frame.iloc[idx,0]
        spkr, emo, fnum = fname.strip().split('_')
        wav_name = os.path.join(self.root_dir, spkr, emo, 'wav_22', ("{:05d}".format(int(fnum)) + '.wav'))
        preprocss_name = os.path.join(hp.preprocess_path, fname)
        mel, mag = get_spectrograms(wav_name)
        
        np.save(preprocss_name + '.pt',  mel)
        np.save(preprocss_name + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
