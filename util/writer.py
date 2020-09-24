import os
from torch.utils.tensorboard import SummaryWriter
from .plot_image import *

#save_fig_dir = './training_log/figures'

def get_writer(output_directory, log_directory):
    logging_path=f'{output_directory}/{log_directory}'
    
    if os.path.exists(logging_path):
        print('The experiment already exists')
        writer = TTSWriter(logging_path)
    else:
        os.mkdir(logging_path)
        writer = TTSWriter(logging_path)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_losses(self, mel_loss, post_mel_loss, global_step, phase):
        self.add_scalar(f'{phase}_mel_loss', mel_loss, global_step)
        self.add_scalar(f'{phase}_post_mel_loss', post_mel_loss, global_step)
        
    def add_alignments(self, enc_alignments, dec_alignments, enc_dec_alignments, style_alignments,
                       mel_lengths, text_lengths, global_step, phase, save_fig_dir):
        enc_align_fig = plot_alignments(enc_alignments, mel_lengths, text_lengths, 'enc')
        self.add_figure(f'{phase}_enc_alignments', enc_align_fig, global_step)
        enc_fig_dir = os.path.join(save_fig_dir, 'enc')
        if not os.path.exists(enc_fig_dir):
            os.makedirs(enc_fig_dir)
        enc_align_fig.savefig(f'{enc_fig_dir}/{global_step}')

        dec_align_fig = plot_alignments(dec_alignments, mel_lengths, text_lengths, 'dec')
        self.add_figure(f'{phase}_dec_alignments', dec_align_fig, global_step)
        dec_fig_dir = os.path.join(save_fig_dir, 'dec')
        if not os.path.exists(dec_fig_dir):
            os.makedirs(dec_fig_dir)
        dec_align_fig.savefig(f'{dec_fig_dir}/{global_step}')

        enc_dec_align_fig = plot_alignments(enc_dec_alignments, mel_lengths, text_lengths, 'enc_dec')
        self.add_figure(f'{phase}_enc_dec_alignments', enc_dec_align_fig, global_step)
        enc_dec_fig_dir = os.path.join(save_fig_dir, 'enc_dec')
        if not os.path.exists(enc_dec_fig_dir):
            os.makedirs(enc_dec_fig_dir)
        enc_dec_align_fig.savefig(f'{enc_dec_fig_dir}/{global_step}')       

        style_align_fig = plot_alignments(style_alignments, mel_lengths, text_lengths, 'style')
        self.add_figure(f'{phase}_style_alignments', style_align_fig, global_step)
        style_fig_dir = os.path.join(save_fig_dir, 'style')
        if not os.path.exists(style_fig_dir):
            os.makedirs(style_fig_dir)
        style_align_fig.savefig(f'{style_fig_dir}/{global_step}')

