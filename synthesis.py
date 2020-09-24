from preprocess import get_dataset, DataLoader, collate_fn_transformer
import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text.HangulUtilsHrim import hangul_to_sequence
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model, ModelStopToken
from collections import OrderedDict
from tqdm import tqdm
import argparse
from util.writer import get_writer

import librosa
import matplotlib
matplotlib.use('Agg')
import librosa.display
import matplotlib.pyplot as plt
import time

import os
import sys

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoints/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def load_waveglow(path):
    waveglow = t.load(path)['model']
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    return waveglow, denoiser.cuda()

def synthesis(args):
    m = Model()
    m_post = ModelPostNet()
    m_stop = ModelStopToken()
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_stop.load_state_dict(load_checkpoint(args.restore_step3, "stop_token"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    m=m.cuda()
    m_post = m_post.cuda()
    m_stop = m_stop.cuda()
    m.train(False)
    m_post.train(False)
    m_stop.train(False)
    test_dataset = get_dataset(hp.test_data_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
    ref_dataset = get_dataset(hp.test_data_csv)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)

    writer = get_writer(hp.checkpoint_path, hp.log_directory)

    ref_dataloader_iter = iter(ref_dataloader)
    for i, data in enumerate(test_dataloader):
        character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, fname = data
        ref_character, ref_mel, ref_mel_input, ref_pos_text, ref_pos_mel, ref_text_length, ref_mel_length, ref_fname = next(ref_dataloader_iter)
        stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)
        mel_input = t.zeros([1,1,80]).cuda()
        stop=[]
        character = character.cuda()
        mel = mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        pos_mel = pos_mel.cuda()
        ref_character = ref_character.cuda()
        ref_mel = ref_mel.cuda()
        ref_mel_input = ref_mel_input.cuda()
        ref_pos_text = ref_pos_text.cuda()
        ref_pos_mel = ref_pos_mel.cuda()

        with t.no_grad():
            start=time.time()
            for i in range(args.max_len):
                pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
                mel_pred, postnet_pred, attn_probs, decoder_output, attns_enc, attns_dec, attns_style = m.forward(character, mel_input, pos_text, pos_mel, ref_mel, ref_pos_mel)
                stop_token = m_stop.forward(decoder_output)
                mel_input = t.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)
                stop.append(t.sigmoid(stop_token).squeeze(-1)[0,-1])
                if stop[-1] > 0.5:
                    print("stop token at " + str(i) + " is :", stop[-1])
                    print("model inference time: ", time.time() - start)
                    break
            if stop[-1] == 0:
                continue
            mag_pred = m_post.forward(postnet_pred)
            inf_time = time.time() - start
            print("inference time: ", inf_time)

        wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
        print("rtx : ", (len(wav)/hp.sr) / inf_time)
        wav_path = os.path.join(hp.sample_path, 'wav')
        if not os.path.exists(wav_path):
            os.makedirs(wav_path)
        write(os.path.join(wav_path, "text_{}_ref_{}_synth.wav".format(fname, ref_fname)), hp.sr, wav)
        print("written as text{}_ref_{}_synth.wav".format(fname, ref_fname))
        attns_enc_new=[]
        attns_dec_new=[]
        attn_probs_new=[]
        attns_style_new=[]
        for i in range(len(attns_enc)):
            attns_enc_new.append(attns_enc[i].unsqueeze(0))
            attns_dec_new.append(attns_dec[i].unsqueeze(0))
            attn_probs_new.append(attn_probs[i].unsqueeze(0))
            attns_style_new.append(attns_style[i].unsqueeze(0))
        attns_enc = t.cat(attns_enc_new, 0)
        attns_dec = t.cat(attns_dec_new, 0)
        attn_probs = t.cat(attn_probs_new, 0)
        attns_style = t.cat(attns_style_new, 0)

        attns_enc = attns_enc.contiguous().view(attns_enc.size(0), 1, hp.n_heads, attns_enc.size(2), attns_enc.size(3))
        attns_enc = attns_enc.permute(1,0,2,3,4)
        attns_dec = attns_dec.contiguous().view(attns_dec.size(0), 1, hp.n_heads, attns_dec.size(2), attns_dec.size(3))
        attns_dec = attns_dec.permute(1,0,2,3,4)
        attn_probs = attn_probs.contiguous().view(attn_probs.size(0), 1, hp.n_heads, attn_probs.size(2), attn_probs.size(3))
        attn_probs = attn_probs.permute(1,0,2,3,4)
        attns_style = attns_style.contiguous().view(attns_style.size(0), 1, hp.n_heads, attns_style.size(2), attns_style.size(3))
        attns_style = attns_style.permute(1,0,2,3,4)

        save_dir = os.path.join(hp.sample_path, 'figure', "text_{}_ref_{}_synth.wav".format(fname, ref_fname))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        writer.add_alignments(attns_enc.detach().cpu(), attns_dec.detach().cpu(), attn_probs.detach().cpu(), attns_style.detach().cpu(), mel_length, text_length, args.restore_step1, 'Validation', save_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=148000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=250000)
    parser.add_argument('--restore_step3', type=int, help='Global step to restore checkpoint', default=8000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=600)

    args = parser.parse_args()
    synthesis(args)
