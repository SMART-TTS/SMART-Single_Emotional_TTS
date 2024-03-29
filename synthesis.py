from preprocess import get_dataset, DataLoader, collate_fn_transformer
import torch as t
from utils import spectrogram2wav, update_kv_mask
from scipy.io.wavfile import write
import hyperparams as hp
from text.HangulUtilsHrim import hangul_to_sequence
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
from util.writer import get_writer
import json

import librosa
import matplotlib
matplotlib.use('Agg')
import librosa.display
import matplotlib.pyplot as plt
import time

import os
import sys

#from mel2audio.args import parse_args
#from mel2audio.hps import Hyperparameters
#from mel2audio.model import SmartVocoder
import soundfile

from hifi_gan.models import Generator
from hifi_gan.env import AttrDict

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoints/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def resample(x, scale, seq_len):
    device = x.device
    batch_size = x.size(0)
    indices = t.arange(2*seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    idx_scaled = indices / scale	# scales must be 0.5~2
    idx_scaled_fl = t.floor(idx_scaled)
    lambda_ = idx_scaled - idx_scaled_fl

    target_mask = idx_scaled_fl < (seq_len-1)
    target_len = target_mask.sum(dim=-1)

    index_1 = t.repeat_interleave(t.arange(batch_size, device=device), target_len)
    idx_2_fl = idx_scaled_fl[target_mask].long()
    idx_2_cl = idx_2_fl + 1
    y_fl = x[index_1, idx_2_fl, :]
    y_cl = x[index_1, idx_2_cl, :]

    lambda_f = lambda_[target_mask].unsqueeze(-1)
    y = (1-lambda_f)*y_fl + lambda_f*y_cl
    return y.unsqueeze(0).repeat(batch_size, 1, 1)
	
def synthesis(args):
    m = Model()
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))   
    m=m.cuda()
    m.train(False)
#    vocoder = SmartVocoder(Hyperparameters(parse_args()))
#    vocoder.load_state_dict(t.load('./mel2audio/merged_STFT_checkpoint.pth')["state_dict"])
#    vocoder=vocoder.cuda()
#    vocoder.eval()
    with open('./hifi_gan/config.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    hifi_gan = Generator(h).cuda()
    state_dict_g = t.load('./hifi_gan/g_00334000', map_location='cuda')
    hifi_gan.load_state_dict(state_dict_g['generator'])
    hifi_gan.eval()
    hifi_gan.remove_weight_norm()


    test_dataset = get_dataset(hp.test_data_csv)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
    ref_dataset = get_dataset(hp.test_data_csv_shuf)
    ref_dataloader = DataLoader(ref_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)

    writer = get_writer(hp.checkpoint_path, hp.log_directory)

    mel_basis = t.from_numpy(librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels, 50, 11000)).unsqueeze(0)  # (n_mels, 1+n_fft//2)
    
    ref_dataloader_iter = iter(ref_dataloader)
    _, ref_mel, _, _, _, ref_pos_mel, _, _, ref_fname = next(ref_dataloader_iter)
   
    for i, data in enumerate(test_dataloader):
        character, _, _, _, pos_text, _, text_length, _, fname = data
        mel_input = t.zeros([1,1,80]).cuda()
        character = character.cuda()
        ref_mel = ref_mel.cuda()
        mel_input = mel_input.cuda()
        pos_text = pos_text.cuda()
        with t.no_grad():
            start=time.time()
            memory, c_mask, attns_enc, duration_mask = m.encoder(character, pos=pos_text)
            style, coarse_emb = m.ref_encoder(ref_mel)
            memory = t.cat((memory, coarse_emb.expand(-1, memory.size(1), -1)), -1)
            memory = m.memory_coarse_layer(memory)
            duration_predictor_output = m.duration_predictor(memory, duration_mask)
            duration = t.ceil(duration_predictor_output)
            duration = duration * duration_mask
#            max_length = t.sum(duration).type(t.LongTensor)
#            print("length : ", max_length)

            monotonic_interpolation, pos_mel_, weights = m.length_regulator(memory, duration, duration_mask)
            kv_mask = t.zeros([1, mel_input.size(1), character.size(1)]).cuda()		# B, t', N
            kv_mask[:, :, :3] = 1
            kv_mask = kv_mask.eq(0)
            stop_flag = False
            ctr = 0
            for j in range(1200):
                pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
                mel_pred, postnet_pred, attn_probs, decoder_output, attns_dec, attns_style = m.decoder(memory, style, mel_input, c_mask,
                                                                                                     pos=pos_mel, ref_pos=ref_pos_mel, mono_inter=monotonic_interpolation[:,:mel_input.shape[1]], kv_mask=kv_mask)
                mel_input = t.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)
#                print("j", j, "mel_input", mel_input.shape)
                if stop_flag and ctr == 10:
                    break
                elif stop_flag:
                    ctr += 1
                kv_mask, stop_flag = update_kv_mask(kv_mask, attn_probs)		# B, t', N --> B, t'+1, N
            postnet_pred = t.cat((postnet_pred, t.zeros(postnet_pred.size(0), 5, postnet_pred.size(-1)).cuda()), 1)
            gen_length = mel_input.size(1)
#            print("gen_length", gen_length)
            post_linear = m.postnet(postnet_pred)
            post_linear = resample(post_linear, seq_len=mel_input.size(1), scale=args.rhythm_scale)
            postnet_pred = resample(mel_input, seq_len=mel_input.size(1), scale=args.rhythm_scale)
            inf_time = time.time() - start
            print("inference time: ", inf_time)
#            print("speech_rate: ", len(postnet_pred[0])/len(character[0]))

            postnet_pred_v = postnet_pred.transpose(2,1)
            postnet_pred_v = (postnet_pred_v*100+20-100)/20
            B, C, T = postnet_pred_v.shape
            z = t.randn(1, 1, T*hp.hop_length).cuda()
            z = z * 0.6 	# Temp
#            t.cuda.synchronize()
#            timestemp = time.time()
#            with t.no_grad():
#                y_gen = vocoder.reverse(z, postnet_pred_v).squeeze()
#            t.cuda.synchronize()
#            print('{} seconds'.format(time.time() - timestemp))
#            wav = y_gen.to(t.device("cpu")).data.numpy()
#            wav = np.pad(wav, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end

            post_linear_v = post_linear.transpose(1,2)
            post_linear_v = 10**((post_linear_v*100+20-100)/20)
            mel_basis = mel_basis.repeat(post_linear_v.shape[0], 1, 1)
            post_linear_mel_v = t.log10(t.bmm(mel_basis.cuda(),post_linear_v))
            B, C, T = post_linear_mel_v.shape
            z = t.randn(1, 1, T*hp.hop_length).cuda()
            z = z * 0.6 	# Temp
#            t.cuda.synchronize()
#            timestemp = time.time()
#            with t.no_grad():
#                y_gen_linear = vocoder.reverse(z, post_linear_mel_v).squeeze()
#            t.cuda.synchronize()
#            wav_linear = y_gen_linear.to(t.device("cpu")).data.numpy()
#            wav_linear = np.pad(wav_linear, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end

            wav_hifi = hifi_gan(post_linear_mel_v).squeeze().clamp(-1,1).detach().cpu().numpy()
            wav_hifi = np.pad(wav_hifi, [0,4800], mode='constant', constant_values=0)		#pad 0 for 0.21 sec silence at the end


        mel_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'mel')
        if not os.path.exists(mel_path):
            os.makedirs(mel_path)
        np.save(os.path.join(mel_path, 'text_{}_ref_{}_synth_{}.mel'.format(i, ref_fname, str(args.rhythm_scale))), postnet_pred.cpu())       

        linear_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'linear')
        if not os.path.exists(linear_path):
            os.makedirs(linear_path)
        np.save(os.path.join(linear_path, 'text_{}_ref_{}_synth_{}.linear'.format(i, ref_fname, str(args.rhythm_scale))), post_linear.cpu())       

#        wav_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav')
#        if not os.path.exists(wav_path):
#            os.makedirs(wav_path)
#        write(os.path.join(wav_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav)
#        print("rtx : ", (len(wav)/hp.sr) / inf_time)

#        wav_linear_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav_linear')
#        if not os.path.exists(wav_linear_path):
#            os.makedirs(wav_linear_path)
#        write(os.path.join(wav_linear_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav_linear)

        wav_hifi_path = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'wav_hifi')
        if not os.path.exists(wav_hifi_path):
            os.makedirs(wav_hifi_path)
        write(os.path.join(wav_hifi_path, "text_{}_ref_{}_synth_{}.wav".format(i, ref_fname, str(args.rhythm_scale))), hp.sr, wav_hifi)  

        show_weights = weights.contiguous().view(weights.size(0), 1, 1, weights.size(1), weights.size(2))
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

        save_dir = os.path.join(hp.sample_path+'_'+str(args.rhythm_scale), 'figure', "text_{}_ref_{}_synth_{}.wav".format(fname, ref_fname, str(args.rhythm_scale)))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        writer.add_alignments(attns_enc.detach().cpu(), attns_dec.detach().cpu(), attn_probs.detach().cpu(), attns_style.detach().cpu(), show_weights.detach().cpu(), [t.tensor(gen_length).type(t.LongTensor)] ,text_length, args.restore_step1, 'Inference', save_dir)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=694000)
    parser.add_argument('--rhythm_scale', type=float, help='Global step to restore checkpoint', default=1.)


    args = parser.parse_args()
    synthesis(args)
