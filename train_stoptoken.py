from preprocess import  DataLoader, get_dataset, collate_fn_transformer 
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm
import sys


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
       
def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoints/checkpoint_%s_%d.pth.tar'% (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def get_mask_from_lengths(lengths):
    max_len = t.max(lengths).item()
    ids = lengths.new_tensor(t.arange(0, max_len))
    mask = (lengths.unsqueeze(1) > ids).to(t.bool)
    return mask

def main():

    dataset = get_dataset(hp.train_data_csv)
    global_step = 0
    
    m = nn.DataParallel(ModelStopToken().cuda())
    trans_model = Model()
    trans_model.load_state_dict(load_checkpoint(100000, "transformer"))
    for name, param in trans_model.named_parameters():
        param.requires_grad = False
        print(name, " : weight frozen")
    trans_model = nn.DataParallel(trans_model.cuda())
   
    m.train()
    trans_model.train(False)


    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    writer = SummaryWriter()

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=8)
        for i, data in enumerate(dataloader):
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            character, mel, mel_input, pos_text, pos_mel, text_length, mel_length, fname = data        
            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()
            mel_length = mel_length.cuda()

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float)-1).cuda()
            for j, length in enumerate(mel_length):
                stop_tokens[j, length-1] += 1

            mel_pred, postnet_pred, attn, decoder_output, _, attn_dec, attn_style = trans_model.forward(character, mel_input, pos_text, pos_mel, mel, pos_mel)
            stop_preds = m.forward(decoder_output)

            if global_step % 100 == 0:
                print("pos_mel", pos_mel[0])
                print("stop_pred", t.sigmoid(stop_preds.squeeze()[0]))
                print("stop_tokens", stop_tokens[0])

            mask = get_mask_from_lengths(mel_length)
            stop_preds = stop_preds.squeeze().masked_select(mask)
            stop_tokens = stop_tokens.masked_select(mask)

            loss = nn.BCEWithLogitsLoss(pos_weight=t.tensor(hp.bce_pos_weight))(stop_preds, stop_tokens)

            print("| Epoch: {}, {}/{}th loss : {:.4f}".format(epoch, i, len(dataloader), loss))

            writer.add_scalars('training_loss',{
                    'loss':loss,

                }, global_step)
                    
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_stop_token_%d.pth.tar' % global_step))

        if epoch == hp.stop_epoch:
            break
    
if __name__ == '__main__':
    main()
