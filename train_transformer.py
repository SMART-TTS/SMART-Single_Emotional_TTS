from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import sys
from utils import spectrogram2wav
from scipy.io.wavfile import write
from util.writer import get_writer
import time
#import torch
from mel2audio.args import parse_args
from mel2audio.hps import Hyperparameters
from mel2audio.model import SmartVocoder
import soundfile
def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def stream(message):
    sys.stdout.write(f"\n{message}")

def get_mask_from_lengths(lengths):
    max_len = t.max(lengths).item()
    ids = lengths.new_tensor(t.arange(0, max_len))
    mask = (lengths.unsqueeze(1) > ids).to(t.bool)
    return mask

def validate(m, vocoder, val_loader, global_step, writer):
    m.eval()
 
    with t.no_grad():
        n_data, val_loss = 0, 0
        for i, data in enumerate(val_loader):
            n_data += len(data[0])
            character, mel, mag, mel_input, pos_text, pos_mel, text_length, mel_length, fname = data

            mel_max_length_array = t.zeros(mel_length.size(0)).long()
            mel_max_length_array = t.LongTensor(mel_max_length_array)
            mel_max_length_array[:] = t.max(mel_length)
            mel_max_length_array = mel_max_length_array.cuda()

            character = character.cuda()
            mel = mel.cuda()
            mag = mag.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            mask = get_mask_from_lengths(mel_length).cuda()

            mel_pred, postnet_pred, attn_probs, decoder_outputs, attns_enc, attns_dec, attns_style, post_linear, duration_predictor_output, duration, weights = m.forward(character, mel_input, pos_text, pos_mel, mel, pos_mel, mel_max_length_array=mel_max_length_array)

            mel_loss = t.mean(t.abs(mel_pred-mel).masked_select(mask.unsqueeze(-1)))
            post_mel_loss = t.mean(t.abs(postnet_pred-mel).masked_select(mask.unsqueeze(-1)))
            n_priority_freq = int(2000 / (hp.sr*0.5) * (hp.n_fft/2+1))
            post_linear_loss = 0.5*t.mean(t.abs(post_linear-mag).masked_select(mask.unsqueeze(-1))) + 0.5*t.mean(t.abs(post_linear-mag)[:, :, :n_priority_freq].masked_select(mask.unsqueeze(-1)))
            duration_loss = nn.L1Loss()(t.sum(duration_predictor_output, -1, keepdim=True), mel_length) / t.sum(text_length)

            loss = mel_loss + post_mel_loss + 0.3*post_linear_loss + duration_loss
            val_loss += loss.item()
        val_loss /= n_data

    for i, mel_pred in enumerate(postnet_pred[:3]):
        mel_v = mel_pred.unsqueeze(0).transpose(2,1)
        mel_v = (mel_v*100+20-100)/20
        B, C, T = mel_v.shape
        z = t.randn(1, 1, T*hp.hop_length).cuda()
        z = z * 0.6     # Temp
        with t.no_grad():
            y_gen = vocoder.reverse(z, mel_v).squeeze()
            wav = y_gen.to(t.device("cpu")).data.numpy()

        mel_path = os.path.join(os.path.join(hp.checkpoint_path, hp.log_directory), 'mel')
        if not os.path.exists(mel_path):
            os.makedirs(mel_path)
        np.save(os.path.join(mel_path, "val_{}_synth_{}.mel".format(fname[i], global_step)), mel_pred.unsqueeze(0).cpu())

        wav_path = os.path.join(os.path.join(hp.checkpoint_path, hp.log_directory), 'wav')
        if not os.path.exists(wav_path):
            os.makedirs(wav_path)
        write(os.path.join(wav_path, "val_{}_synth_{}.wav".format(fname[i], global_step)), hp.sr, wav)
        print("written as val_{}_synth.wav".format(fname[i]))

    for i, mel_ in enumerate(mel[:3]):
        mel_v = mel_.unsqueeze(0).transpose(2,1)
        mel_v = (mel_v*100+20-100)/20
        B, C, T = mel_v.shape
        z = t.randn(1, 1, T*hp.hop_length).cuda()
        z = z * 0.6     # Temp
        with t.no_grad():
            y_gen = vocoder.reverse(z, mel_v).squeeze()
            wav = y_gen.to(t.device("cpu")).data.numpy()

        mel_path = os.path.join(os.path.join(hp.checkpoint_path, hp.log_directory), 'mel_recon')
        if not os.path.exists(mel_path):
            os.makedirs(mel_path)
        np.save(os.path.join(mel_path, "val_{}_synth_{}.mel".format(fname[i], global_step)), mel_.unsqueeze(0).cpu())

        wav_path = os.path.join(os.path.join(hp.checkpoint_path, hp.log_directory), 'wav_recon')
        if not os.path.exists(wav_path):
            os.makedirs(wav_path)
        write(os.path.join(wav_path, "val_{}_synth_{}.wav".format(fname[i], global_step)), hp.sr, wav)
        print("written as val_{}_synth.wav".format(fname[i]))

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

    attns_enc = attns_enc.contiguous().view(attns_enc.size(0), hp.batch_size, hp.n_heads, attns_enc.size(2), attns_enc.size(3))
    attns_enc = attns_enc.permute(1,0,2,3,4)
    attns_dec = attns_dec.contiguous().view(attns_dec.size(0), hp.batch_size, hp.n_heads, attns_dec.size(2), attns_dec.size(3))
    attns_dec = attns_dec.permute(1,0,2,3,4)
    attn_probs = attn_probs.contiguous().view(attn_probs.size(0), hp.batch_size, hp.n_heads, attn_probs.size(2), attn_probs.size(3))
    attn_probs = attn_probs.permute(1,0,2,3,4)
    attns_style = attns_style.contiguous().view(attns_style.size(0), hp.batch_size, hp.n_heads, attns_style.size(2), attns_style.size(3))
    attns_style = attns_style.permute(1,0,2,3,4)

    save_dir = os.path.join(hp.checkpoint_path, hp.log_directory, 'figure')
    writer.add_losses(mel_loss.item(), post_mel_loss.item(), 0.3*post_linear_loss.item(), duration_loss.item(), global_step, 'Validation')
    writer.add_alignments(attns_enc.detach().cpu(), attns_dec.detach().cpu(), attn_probs.detach().cpu(), attns_style.detach().cpu(), mel_length, text_length, global_step, 'Validation', save_dir)

    msg = "Validation| loss : {:.4f} + {:.4f} + {:.4f} + {:.4f} = {:.4f}".format(mel_loss, post_mel_loss, 0.3*post_linear_loss, duration_loss, loss)
    stream(msg)
    m.train()


def main():

    train_dataset = get_dataset(hp.train_data_csv)
    val_dataset = get_dataset(hp.val_data_csv)
    restore_step = hp.restore_step
    global_step = restore_step
    if restore_step != 0:
        restore_flag = True
    else:
        restore_flag = False

    m = Model()
    if os.path.exists('./checkpoints/checkpoint_%s_%d.pth.tar'% ('transformer', global_step)):
        state_dict = t.load('./checkpoints/checkpoint_%s_%d.pth.tar'% ('transformer', global_step))   
        new_state_dict = OrderedDict()
        for k, value in state_dict['model'].items():
            key = k[7:]
            new_state_dict[key] = value
    
        m.load_state_dict(new_state_dict)

    m = nn.DataParallel(m.cuda())
    m.train()

    vocoder = SmartVocoder(Hyperparameters(parse_args()))
    vocoder.load_state_dict(t.load('./mel2audio/checkpoint_step000588458.pth')["state_dict"])
    vocoder=vocoder.cuda()
    vocoder.eval()

    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    writer = get_writer(hp.checkpoint_path, hp.log_directory)
    cur_epoch = 0

    for epochs in range(hp.epochs):
        train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last = True)
        if restore_flag:
            cur_epoch = int(restore_step/len(train_dataloader))
            restore_flag = not restore_flag
        for i, data in enumerate(train_dataloader):
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mag, mel_input, pos_text, pos_mel, text_length, mel_length, fname =  data

            mel_max_length_array = t.zeros(mel_length.size(0)).long()
            mel_max_length_array = t.LongTensor(mel_max_length_array)
            mel_max_length_array[:] = t.max(mel_length)
            mel_max_length_array = mel_max_length_array.cuda()

            character = character.cuda()
            mel = mel.cuda()
            mag = mag.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            loading_time = time.time()
            mask = get_mask_from_lengths(mel_length).cuda()

            mel_pred, postnet_pred, attn_probs, decoder_outputs, attns_enc, attns_dec, attns_style, post_linear, duration_predictor_output, duration, weights = m.forward(character, mel_input, pos_text, pos_mel, mel, pos_mel, mel_max_length_array=mel_max_length_array)

            mel_loss = t.mean(t.abs(mel_pred-mel).masked_select(mask.unsqueeze(-1)))
            post_mel_loss = t.mean(t.abs(postnet_pred- mel).masked_select(mask.unsqueeze(-1)))
            n_priority_freq = int(2000 / (hp.sr*0.5) * (hp.n_fft/2+1))
            post_linear_loss = 0.5*t.mean(t.abs(post_linear-mag).masked_select(mask.unsqueeze(-1))) + 0.5*t.mean(t.abs(post_linear-mag)[:, :, :n_priority_freq].masked_select(mask.unsqueeze(-1)))
            duration_loss = nn.L1Loss()(t.sum(duration_predictor_output, -1, keepdim=True), mel_length) / t.sum(text_length)

            loss = (mel_loss + post_mel_loss + 0.3*post_linear_loss + duration_loss)/hp.accum
            writer.add_losses(mel_loss.item(), post_mel_loss.item(), 0.3*post_linear_loss, duration_loss, global_step, 'Train')

            # Calculate gradients
            loss.backward()
            msg = "| Epoch: {}, {}/{}th loss : {:.4f} + {:.4f} + {:.4f} + {:.4f} = {:.4f}".format(cur_epoch, i, len(train_dataloader), mel_loss, post_mel_loss, 0.3*post_linear_loss, duration_loss, loss)
            stream(msg)

            if global_step % hp.accum == 0:
                nn.utils.clip_grad_norm_(m.parameters(), 1.)        
                # Update weights
                optimizer.step()
                optimizer.zero_grad()

            if global_step % hp.val_step == 0 or global_step==1:
                validate(m, vocoder, val_dataloader, global_step, writer)

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))
        if cur_epoch == hp.stop_epoch:
            break
        cur_epoch += 1
        print(' ')
            
            


if __name__ == '__main__':
    main()
