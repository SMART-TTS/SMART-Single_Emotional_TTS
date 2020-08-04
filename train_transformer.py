from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from module import GuidedLoss 
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


def validate(m, val_loader, global_step, writer):
    m.eval()
    with t.no_grad():
        n_data, val_loss = 0, 0
        for i, data in enumerate(val_loader):
            n_data += len(data[0])
            character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, emo_code = data
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1).cuda()
            
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel, emo_code)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            # stop_token_loss = nn.BCEWithLogitsLoss(pos_weight=t.tensor(hp.bce_pos_weight))(stop_preds, stop_tokens)
            # guide_loss = guided_loss(attn_probs, text_length, mel_length, global_step) # isn't it correct to write guide?

            # loss = mel_loss + post_mel_loss + guide_loss
            loss = mel_loss + post_mel_loss
            val_loss += loss.item()
        val_loss /= n_data
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
    if os.path.exists('./checkpoints_guide/checkpoint_%s_%d.pth.tar'% ('transformer', global_step)):
        state_dict = t.load('./checkpoints_guide/checkpoint_%s_%d.pth.tar'% ('transformer', global_step))
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
        val_dataloader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True)
        if restore_flag:
            cur_epoch = int(restore_step/len(train_dataloader))
            restore_flag = not restore_flag
        # pbar = tqdm(train_dataloader)
        for i, data in enumerate(train_dataloader):
            # start = time.time()
            # pbar.set_description("Processing at epoch %d"%cur_epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, emo_code = data
            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1).cuda()	# [0,0,...,0,0,1,1,1,...,1,1]
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            text_length = text_length.cuda()
            mel_length = mel_length.cuda()
            loading_time = time.time()
            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel, emo_code)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)
            # stop_token_loss = nn.BCEWithLogitsLoss(pos_weight=t.tensor(hp.bce_pos_weight))(stop_preds, stop_tokens)
            guide = GuidedLoss()
            guide_loss = guide(attn_probs, text_length, mel_length, global_step)

            # if epochs % 10 and epochs != 0:
            #     loss = (mel_loss + post_mel_loss + 0.1*stop_token_loss + guide_loss)/hp.accum
            # else:
            loss = (mel_loss + post_mel_loss + guide_loss)/hp.accum
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
                t.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict()},
                       os.path.join(hp.checkpoint_path,'checkpoint_transformer_%d.pth.tar' % global_step))
        cur_epoch += 1
        print(' ')


if __name__ == '__main__':
    main()
