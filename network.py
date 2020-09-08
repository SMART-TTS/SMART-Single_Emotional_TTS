from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparams as hp
import copy

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
#        x = pos * self.alpha + x

        # Positional dropout
#        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x = pos * self.alpha + x
            x = self.pos_dropout(x)
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask, attns


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
        self.norm = Linear(q_num_hidden, q_num_hidden)

        self.selfattn_layers = clones(Attention(q_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.styleattn_layers = clones(Attention(style_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.dotattn_layers = clones(Attention(kv_num_hidden, q_num_hidden, num_hidden, hp.n_heads), hp.n_layers)
        self.ffns = clones(FFN(q_num_hidden), hp.n_layers)
        self.mel_linear = Linear(q_num_hidden, hp.num_mels * hp.outputs_per_step)
#        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')

        self.postconvnet = PostConvNet(q_num_hidden)

    def forward(self, memory, style, decoder_input, c_mask, pos, ref_pos):
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

        for selfattn, styleattn, dotattn, ffn in zip(self.selfattn_layers, self.styleattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
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

        self.downsample = Conv(in_channels=hp.style_size*2,
	                        out_channels=hp.style_size*2,
                            kernel_size=hp.downsample_kernel_size,
                            stride=hp.downsample_stride,
                            padding=int(np.floor(hp.downsample_kernel_size/hp.downsample_stride)),
                            w_init='relu')
       
    def forward(self, emb):
       inter_emb = emb.contiguous().transpose(1,2)
       inter_emb = self.dropout(self.layer_norm1(t.relu(self.conv1(inter_emb)).contiguous().transpose(1,2))).transpose(1,2)
       inter_emb = self.dropout(self.layer_norm2(t.relu(self.conv2(inter_emb)).contiguous().transpose(1,2))).transpose(1,2)
       inter_emb = self.dropout(self.layer_norm3(t.relu(self.conv3(inter_emb)).contiguous().transpose(1,2)))
       out_emb, last_state_emb = self.gru(inter_emb)
#       out_emb = t.mean(out_emb, 1, keepdim=True)
       ref_logit = t.relu(self.linear1(out_emb))
       ref_logit = t.relu(self.linear2(ref_logit))
#       ref_logit = self.downsample(ref_logit)
#       print("ref_logit", ref_logit.shape)
       
       return ref_logit


class ModelStopToken(nn.Module):
    def __init__(self):
        super(ModelStopToken, self).__init__()
        self.linear1 = Linear(hp.hidden_size, 2*hp.hidden_size, w_init='sigmoid')
        self.stop_linear = Linear(2*hp.hidden_size, 1, w_init='sigmoid')

    def forward(self, decoder_output):
        # Stop tokens
        stop_tokens = self.stop_linear(t.relu(self.linear1(decoder_output)))
        return stop_tokens

class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder(hp.embedding_size, hp.hidden_size)	# embedding size = 512, hidden size = 256
        self.ref_encoder = RefEncoder()
        self.decoder = MelDecoder(hp.text_hidden_size, hp.style_size, hp.history_hidden_size, hp.hidden_size)
        self.num_params()

    def forward(self, characters, mel_input, pos_text, pos_mel, ref_mel, ref_pos_mel):
        memory, c_mask, attns_enc = self.encoder(characters, pos=pos_text)
        style = self.ref_encoder(ref_mel)
        mel_output, postnet_output, attn_probs, decoder_output, attns_dec, attns_style = self.decoder(memory, style, mel_input, c_mask,
                                                                                             pos=pos_mel, ref_pos=ref_pos_mel)

        return mel_output, postnet_output, attn_probs, decoder_output, attns_enc, attns_dec, attns_style

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

class ModelPostNet(nn.Module):
    """
    CBHG Network (mel --> linear)
    """
    def __init__(self):
        super(ModelPostNet, self).__init__()
        self.pre_projection = Conv(hp.n_mels, hp.hidden_size)
        self.cbhg = CBHG(hp.hidden_size)
        self.post_projection = Conv(hp.hidden_size, (hp.n_fft // 2) + 1)
        self.num_params()

    def forward(self, mel):
        mel = mel.transpose(1, 2)
        mel = self.pre_projection(mel)
        mel = self.cbhg(mel).transpose(1, 2)
        mag_pred = self.post_projection(mel).transpose(1, 2)
        return mag_pred

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)


