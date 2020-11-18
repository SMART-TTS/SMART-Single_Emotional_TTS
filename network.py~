from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparams as hp
import copy

# add length regulator
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def length_regul(self, x, duration_predictor_output, duration_mask, mel_max_length=0):
        expand_max_len = t.max(t.sum(duration_predictor_output, -1), -1)[0]
        if mel_max_length != 0:  # for training and validation for transformer and duration predictor
            expand_max_len = mel_max_length
            pos_mel = t.stack(
                [t.stack([t.Tensor([i + 1 for i in range(expand_max_len)])]).long() for j in
                 range(duration_predictor_output.size(0))],
                dim=0).squeeze().cuda()  # (batch, seq_mel)
        else:  # for inference
            expand_max_len = (expand_max_len+1000).long()   # +100 frames for synthesis for sad
            pos_mel = t.stack([t.Tensor([i + 1 for i in range(expand_max_len)])]).long().squeeze().cu    da()  # (batch, seq_mel)
            pos_mel = t.unsqueeze(pos_mel, 0)

        centre = t.ceil(t.cumsum(duration_predictor_output, dim=-1) - t.ceil(0.5 * duration_predictor    _output))  # (batch, seq_char)
        sigma_square = 10.0
        centre_diff = t.unsqueeze(centre, 1).expand(-1, expand_max_len, -1)
        pos_mel_diff = t.unsqueeze(pos_mel, 2).expand(-1, -1, duration_predictor_output.size(1))

        diff = centre_diff - pos_mel_diff
        logits = -(diff**2/sigma_square)  # (batch, T=seq_mel, L=seq_char)

        logits_inv_mask = 1. - t.unsqueeze(duration_mask, 1).expand(-1, expand_max_len, -1)

        masked_logits = logits - 1e9 * logits_inv_mask
        weights = t.nn.Softmax(dim=2)(masked_logits)  # (batch, T, L)
        output = t.bmm(weights, x)  # (batch, T, L) x (batch, L, num_hidden) = (batch, T, num_hidden)

        return output, pos_mel, weights

    def forward(self, x, duration_predictor_output, duration_mask, mel_max_length = None):  # x = enc    oder_output
        if mel_max_length is not None:  # for training & validation
            output, pos_mel, weights = self.length_regul(x, duration_predictor_output, duration_mask,     mel_max_length)
        else:  # for inference
            output, pos_mel, weights = self.length_regul(x, duration_predictor_output, duration_mask)      # output: (batch, seq_mel_pred, hp.hidden_size)
        # pos_mel is necessary when you do positional encoding to decoder at inference for Non-Autore    gressive model
        return output, pos_mel, weights

class Duration_Linear(nn.Module):
    def __init__(self):
        super(Duration_Linear, self).__init__()
        self.linear1 = Linear(hp.hidden_size, 2*hp.hidden_size)
        self.stop_linear = Linear(2*hp.hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output, duration_mask):
        # duration
        duration = self.stop_linear.forward(self.linear1.forward(encoder_output))
        duration = self.relu(duration)
        duration = t.squeeze(duration)
        duration = duration * duration_mask
        return duration



class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: dimension of embedding	512
        :param num_hidden: dimension of hidden	256
        """
        super(Encoder, self).__init__()
        self.alpha = nn.Parameter(t.ones(1))	#1
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)		#[1024, 256]
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)		# output:256
        self.layers = clones(Attention(num_hidden, num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.ffns = clones(FFN(num_hidden), hp.n_layers)

    def forward(self, x, pos):
        duration_mask = pos.ne(0).type(t.float)
        batch_size = x.size(0)
        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(t.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        # Encoder pre-network
        x = self.encoder_prenet(x)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x = pos * self.alpha + x
            x = self.pos_dropout(x)
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask, attns, duration_mask


class MelDecoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self, kv_num_hidden, style_num_hidden, q_num_hidden, num_hidden):
        """
        :param num_hidden: dimension of hidden
        """
        super(MelDecoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        self.decoder_prenet = Prenet(hp.num_mels, num_hidden * 2, num_hidden, p=0.2)
        self.mono_prenet = Prenet(num_hidden, num_hidden * 2, num_hidden, p=0.2)
        self.decoder_mono_prenet = Prenet(num_hidden, num_hidden * 2, num_hidden, p=0.2)

        self.norm = Linear(q_num_hidden, q_num_hidden)

        self.selfattn_layers = clones(Attention(q_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.styleattn_layers = clones(Attention(style_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.dotattn_layers = clones(Attention(kv_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.ffns = clones(FFN(q_num_hidden), hp.n_layers)
        self.mel_linear = Linear(q_num_hidden, hp.num_mels * hp.outputs_per_step)

        self.postconvnet = PostConvNet(q_num_hidden)

    def forward(self, memory, style, decoder_input, c_mask, pos, ref_pos, kv_mask=None):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)
        ref_len = style.size(1)

        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(t.float)
            style_mask = ref_pos.ne(0).type(t.float)
            style_mask = style_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            style_mask = style_mask.transpose(1,2)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            mask = t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask, style_mask = None, None, None

        # Decoder pre-network
        decoder_input = self.decoder_prenet(decoder_input)

        mono_inter = self.mono_prenet(mono_inter)
        decoder_input = decoder_input + mono_inter
        decoder_input = self.decoder_mono_prenet(decoder_input)

        # Centered position
        decoder_input = self.norm(decoder_input)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)

        decoder_input = pos * self.alpha + decoder_input

        # Positional dropout
        decoder_input = self.pos_dropout(decoder_input)

        # Attention decoder-decoder, encoder-decoder
        attn_dot_list = list()
        attn_style_list = list()
        attn_dec_list = list()

        for idx, (selfattn, styleattn, dotattn, ffn) in enumerate(zip(self.selfattn_layers, self.styleattn_layers, self.dotattn_layers, self.ffns)):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            if idx == 0:
                decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask, kv_mask=kv_mask)
            else:
                decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)

             decoder_input, attn_style = styleattn(style, decoder_input, mask=style_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_style_list.append(attn_style)
            attn_dec_list.append(attn_dec)
        # Mel linear projection
        mel_out = self.mel_linear(decoder_input)
        
        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)

        return mel_out, out, attn_dot_list, decoder_input, attn_dec_list, attn_style_list


class RefEncoder(nn.Module):
    def __init__(self):
        super(RefEncoder, self).__init__()
        self.conv1 = Conv(in_channels=hp.n_mels,
                            out_channels=hp.ref_filter_size1,
                            kernel_size=hp.ref_kernel_size,
                            padding=int(np.floor(hp.ref_kernel_size/2)),
                            w_init='relu')
        self.layer_norm1 = nn.LayerNorm(hp.ref_filter_size1)
        self.conv2 = Conv(in_channels=hp.ref_filter_size1,
                            out_channels=hp.ref_filter_size2,
                            kernel_size=hp.ref_kernel_size,
                            padding=int(np.floor(hp.ref_kernel_size/2)),
                            w_init='relu')
        self.layer_norm2 = nn.LayerNorm(hp.ref_filter_size2)
        self.conv3 = Conv(in_channels=hp.ref_filter_size2,
                            out_channels=hp.ref_filter_size3,
                            kernel_size=hp.ref_kernel_size,
                            padding=int(np.floor(hp.ref_kernel_size/2)),
                            w_init='relu')
        self.layer_norm3 = nn.LayerNorm(hp.ref_filter_size3)

        self.gru = nn.GRU(input_size=hp.ref_filter_size3, hidden_size=hp.ref_gru_width,
                            num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = Linear(hp.ref_gru_width*2, hp.ref_gru_width)
        self.linear2 = Linear(hp.ref_gru_width, hp.style_size)
        self.coarse_layer = Linear(hp.ref_gru_width*2, hp.coarse_size)
       
    def forward(self, emb):
       inter_emb = emb.contiguous().transpose(1,2)
       inter_emb = self.dropout(self.layer_norm1(t.relu(self.conv1(inter_emb)).contiguous().transpose(1,2))).transpose(1,2)
       inter_emb = self.dropout(self.layer_norm2(t.relu(self.conv2(inter_emb)).contiguous().transpose(1,2))).transpose(1,2)
       inter_emb = self.dropout(self.layer_norm3(t.relu(self.conv3(inter_emb)).contiguous().transpose(1,2)))
       out_emb, last_state_emb = self.gru(inter_emb)
       coarse_emb = t.mean(out_emb, 1, keepdim=True)
       coarse_emb = self.coarse_layer(coarse_emb)
       ref_logit = t.relu(self.linear1(out_emb))
       ref_logit = t.relu(self.linear2(ref_logit))
      
       return ref_logit, coarse_emb

class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)
#        self.num_params()

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)
        return mag_pred


class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)	# embedding size = 512, hidden size = 256
        self.ref_encoder = RefEncoder()
        self.decoder = MelDecoder(hp.text_hidden_size, hp.style_size, hp.history_hidden_size, hp.hidden_size)

        self.postnet = ModelPostNet()
        self.memory_coarse_layer = Linear(hp.coarse_size + hp.hidden_size, hp.hidden_size)
        self.duration_predictor = Duration_Linear()
        self.length_regulator = LengthRegulator()
        self.num_params()


    def forward(self, characters, mel_input, pos_text, pos_mel, ref_mel, ref_pos_mel, mel_max_length_    array=None):
        memory, c_mask, attns_enc, duration_mask = self.encoder(characters, pos=pos_text)
        style, coarse_emb = self.ref_encoder(ref_mel)
        memory = t.cat((memory, coarse_emb.expand(-1, memory.size(1), -1)), -1)
        memory = self.memory_coarse_layer(memory)

        duration_predictor_output = self.duration_predictor.forward(memory.clone().detach(), duration    _mask)
        duration = t.ceil(duration_predictor_output.clone().detach())  # this make gradient to zero
        duration = duration * duration_mask

        if mel_max_length_array is not None:
            mel_max_length = mel_max_length_array[0]
            monotonic_interpolation, pos_mel_, weights = self.length_regulator(memory, duration, dura    tion_mask, mel_max_length=mel_max_length)
            mel_output, postnet_output, attn_probs, decoder_output, attns_dec, attns_style = self.dec    oder(memory, style, mel_input, c_mask,
                                                                                             pos=pos_    mel, ref_pos=ref_pos_mel, mono_inter=monotonic_interpolation[:,:mel_input.shape[1]])
        else:
            monotonic_interpolation, pos_mel_, weights = self.length_regulator(memory, duration, dura    tion_mask)
            mel_output, postnet_output, attn_probs, decoder_output, attns_dec, attns_style = self.dec    oder(memory, style, mel_input, c_mask,
                                                                                             pos=pos_    mel, ref_pos=ref_pos_mel, mono_inter=monotonic_interpolation[:,:mel_input.shape[1]])

        post_linear = self.postnet(postnet_output)
        return mel_output, postnet_output, attn_probs, decoder_output, attns_enc, attns_dec, attns_st    yle, post_linear, duration_predictor_output, duration, weights

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)


