from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import sys
#from util.utils import * 
from util.writer import get_writer
import time
#import torch
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

def validate(m, val_loader, global_step, writer):
    m.eval()
    with t.no_grad():
        n_data, val_loss = 0, 0
        for i, data in enumerate(val_loader):
            n_data += len(data[0])
            character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, fname = data
           
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            
            mel_pred, postnet_pred, attn_probs, decoder_outputs, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            mask = get_mask_from_lengths(mel_length).cuda()
   
            loss = mel_loss + post_mel_loss
            val_loss += loss.item()
        val_loss /= n_data
    attns_enc_new=[]
    attns_dec_new=[]
    attn_probs_new=[]
    for i in range(len(attns_enc)):
        attns_enc_new.append(attns_enc[i].unsqueeze(0))
        attns_dec_new.append(attns_dec[i].unsqueeze(0))
        attn_probs_new.append(attn_probs[i].unsqueeze(0))
    attns_enc = t.cat(attns_enc_new, 0)
    attns_dec = t.cat(attns_dec_new, 0)
    attn_probs = t.cat(attn_probs_new, 0)

    attns_enc = attns_enc.contiguous().view(attns_enc.size(0), hp.batch_size, hp.n_heads, attns_enc.size(2), attns_enc.size(3))
    attns_enc = attns_enc.permute(1,0,2,3,4)
    attns_dec = attns_dec.contiguous().view(attns_dec.size(0), hp.batch_size, hp.n_heads, attns_dec.size(2), attns_dec.size(3))
    attns_dec = attns_dec.permute(1,0,2,3,4)
    attn_probs = attn_probs.contiguous().view(attn_probs.size(0), hp.batch_size, hp.n_heads, attn_probs.size(2), attn_probs.size(3))
    attn_probs = attn_probs.permute(1,0,2,3,4)
    save_dir = os.path.join(hp.checkpoint_path, hp.log_directory, 'figure')
    writer.add_losses(mel_loss.item(), post_mel_loss.item(), global_step, 'Validation')
    writer.add_alignments(attns_enc.detach().cpu(), attns_dec.detach().cpu(), attn_probs.detach().cpu(), mel_length, text_length, global_step, 'Validation', save_dir)

    msg = "Validation| loss : {:.4f} + {:.4f} = {:.4f}".format(mel_loss, post_mel_loss, loss)
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
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)


    pos_weight = t.FloatTensor([5.]).cuda()
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
                
            character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, fname =  data
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            loading_time = time.time()
            mel_pred, postnet_pred, attn_probs, decoder_output, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)
            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            loss = (mel_loss + post_mel_loss)/hp.accum
            writer.add_losses(mel_loss.item(), post_mel_loss.item(), global_step, 'Train')

            # Calculate gradients
            loss.backward()
            msg = "| Epoch: {}, {}/{}th loss : {:.4f} + {:.4f} = {:.4f}".format(cur_epoch, i, len(train_dataloader), mel_loss, post_mel_loss, loss)
            stream(msg)

            if global_step % hp.accum == 0:
                nn.utils.clip_grad_norm_(m.parameters(), 1.)        
                # Update weights
                optimizer.step()
                optimizer.zero_grad()

            if global_step % hp.val_step == 0 or global_step==1:
                validate(m, val_dataloader, global_step, writer)

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
