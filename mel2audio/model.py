import torch
from torch import nn
from math import log, pi, sqrt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal


def tanh_sigmoid_activation(n_channels, input_a, input_b, input_c=None):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b 
    if input_c is not None:
        in_act = in_act + input_c 
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act

    return acts


class WaveNet(nn.Module):
    def __init__(self, in_channels, cin_channels, di_cycle, pos_group, n_channels, n_layers, kernel_size=3):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.pos_group = pos_group

        self.in_layers = torch.nn.ModuleList()
        self.time_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        out_channels = in_channels * 2
        end = torch.nn.Conv1d(n_channels, out_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        cond_layer = torch.nn.Conv1d(cin_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        
        if pos_group > 1:
            pos_emb = torch.nn.Embedding(pos_group, 2*n_channels*n_layers)
            self.pos_emb = torch.nn.utils.weight_norm(pos_emb, name='weight')

        for i in range(n_layers):
            dilation = 2 ** (i % di_cycle)
            padding = int((kernel_size*dilation - dilation)) // 2

            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio, spect, pos=None):
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        if pos is not None:
            pos = self.pos_emb(pos)
            pos = pos.unsqueeze(2)

        for i in range(self.n_layers):
            pos_offset = spect_offset = i*2*self.n_channels
            spect_in = spect[:, spect_offset:spect_offset+2*self.n_channels, :]
            pos_in = pos[:, pos_offset:pos_offset+2*self.n_channels, :] if pos is not None else None

            acts = tanh_sigmoid_activation(n_channels_tensor, self.in_layers[i](audio), spect_in, pos_in)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
        
        return self.end(output).chunk(2,1)


class SqueezeLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x = self.squeeze(x, self.scale)

        return x

    def reverse(self, z):
        z = self.unsqueeze(z, self.scale)

        return z

    def squeeze(self, x, scale):
        B, C, T = x.size()
        squeezed_x = x.contiguous().view(B, C, T // scale, scale).permute(0, 1, 3, 2)
        squeezed_x = squeezed_x.contiguous().view(B, C * scale, T // scale)

        return squeezed_x

    def unsqueeze(self, z, scale):
        B, C, T = z.size()
        unsqueezed_z = z.view(B, C // scale, scale, T).permute(0, 1, 3, 2)
        unsqueezed_z = unsqueezed_z.contiguous().view(B, C // scale, T * scale)

        return unsqueezed_z


class Invertible1x1Conv(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, x, c, log_det_sum):
        # shape
        B_, _, T_ = x.size()
        W = self.conv.weight.squeeze()
        z = self.conv(x)
        log_det_W = B_ * T_ * torch.logdet(W)
        log_det_sum = log_det_sum + log_det_W

        return z, c, log_det_sum

    def reverse(self, z, c):
        W = self.conv.weight.squeeze()
        W_inverse = W.float().inverse()
        W_inverse = Variable(W_inverse[..., None])
        self.W_inverse = W_inverse
        x = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        
        return x, c


class ActNorm(nn.Module):
    def __init__(self, in_channels, pretrained):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1))

        self.initialized = pretrained

    def initialize(self, x):
        flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        mean = (
            flatten.mean(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .permute(1, 0, 2)
        )
        std = (
            flatten.std(1)
            .unsqueeze(1)
            .unsqueeze(2)
            .permute(1, 0, 2)
        )

        self.loc.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x, c, log_det_sum):
        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        z = self.scale * (x + self.loc)

        log_abs = torch.log(torch.abs(self.scale))
        B_, _, T_ = x.size()
        log_det_sum = log_det_sum + (log_abs.sum() * B_ * T_)

        return z, c, log_det_sum

    def reverse(self, z, c):
        x = (z / self.scale) - self.loc

        return x, c


class PosConditionedFlow(nn.Module):
    def __init__(self, in_channels, cin_channels, dilation, pos_group, n_channels, n_layers):
        super().__init__()
        self.pos_group = pos_group
        self.in_channels = in_channels
        self.WN = WaveNet(in_channels//2, cin_channels, dilation, pos_group, n_channels, n_layers)

    def forward(self, x, c, log_det_sum):
        if self.pos_group > 1:
            B_orig = x.shape[0] // self.pos_group
            pos = torch.tensor(range(self.pos_group)).to(x.device).repeat(B_orig)
        else:
            pos = None

        x_a, x_b = x.chunk(2,1)
        log_s, b = self.WN(x_a, c, pos)
        x_b = torch.exp(log_s) * x_b + b
        log_det_sum = log_det_sum + log_s.sum()
        z = torch.cat((x_a, x_b), dim=1)

        return z, c, log_det_sum

    def reverse(self, z, c):
        if self.pos_group > 1:
            B_orig = z.shape[0] // self.pos_group
            pos = torch.tensor(range(self.pos_group)).to(z.device).repeat(B_orig)
        else:
            pos = None

        z_a, z_b = z.chunk(2,1)
        log_s, b = self.WN(z_a, c, pos)
        z_b = torch.exp(-log_s) * (z_b - b)

        x = torch.cat((z_a, z_b), dim=1)

        return x, c


class EqualResolutionBlock(nn.Module):
    def __init__(self, chains):
        super().__init__()
        self.chains = nn.ModuleList(chains)

    def forward(self, x, c, log_det_sum):
        for chain in self.chains:
            x, c, log_det_sum = chain(x, c, log_det_sum)
        z = x

        return z, log_det_sum

    def reverse(self, z, c):
        for chain in self.chains[::-1]:
            z, c = chain.reverse(z, c)
        x = z

        return x, c


class UpsampleConv(nn.Module):
    def __init__(self, sc_i, sc, hl, n_blocks):
        super().__init__()
        self.conv_list = nn.ModuleList()

        up_list = [hl // (sc_i * (sc ** (n_blocks-1)))] + [sc for _ in range(n_blocks - 1)]

        for u in up_list:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * u), padding=(1, u // 2), stride=(1, u))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.conv_list.append(convt)
            self.conv_list.append(nn.LeakyReLU(0.4))

    def forward(self, mel):
        c_list = []
        c = mel.unsqueeze(1)
        for layer in self.conv_list:
            c = layer(c)
            if isinstance(layer, nn.ConvTranspose2d) and layer.stride[1] % 2 == 1:
                c = c[:,:, :, :-1]
            elif isinstance(layer, nn.LeakyReLU):
                c_list.append(c.squeeze(1))

        return c_list


class SmartVocoder(nn.Module):
    def __init__(self, hps):
        super().__init__()

        in_channels = 1  # number of channels in audio
        cin_channels = 80  # number of channels in mel-spectrogram (freq. axis)
        self.sqz_scale_i = hps.sqz_scale_i
        self.sqz_scale = hps.sqz_scale

        self.n_ER_blocks = hps.n_ER_blocks
        self.n_flow_blocks = hps.n_flow_blocks
        self.n_layers = hps.n_layers
        self.n_channels = hps.n_channels
        self.pretrained = hps.pretrained
        self.sqz_layer = SqueezeLayer(hps.sqz_scale_i)
        self.ER_blocks = nn.ModuleList()
        self.upsample_conv = UpsampleConv(hps.sqz_scale_i, hps.sqz_scale, hps.hop_length, hps.n_ER_blocks)

        in_channels *= hps.sqz_scale_i
        pos_group = 1
        for i in range(hps.n_ER_blocks):
            dilation_cycle = hps.di_cycle[i]
            self.ER_blocks.append(self.build_ER_block(hps.n_flow_blocks, in_channels, cin_channels, dilation_cycle, 
                                pos_group, hps.n_channels, hps.n_layers[i], hps.pretrained))
            pos_group *= hps.sqz_scale
        
    def build_ER_block(self, n_flow_blocks, in_channels, cin_channels, di_cycle, pos_group, n_channels, n_layers, pretrained):
        chains = []
        for _ in range(n_flow_blocks):
            chains += [ActNorm(in_channels, pretrained=pretrained)]
            chains += [Invertible1x1Conv(in_channels)]
            chains += [PosConditionedFlow(in_channels, cin_channels, di_cycle, pos_group, n_channels, n_layers)]

        ER_block = EqualResolutionBlock(chains)

        return ER_block

    def forward(self, x, mel):
        Bx, Cx, Tx = x.size()
        sc = self.sqz_scale

        c_list = self.upsample_conv(mel)
        c_list = c_list[::-1]

        out = self.sqz_layer(x)
        log_det_sum = 0.0
        c_in = c_list[0]
        for i, block in enumerate(self.ER_blocks):
            out, log_det_sum = block(out, c_in, log_det_sum)

            if i != len(self.ER_blocks) -1:
                B, C, T = out.shape
                out = out.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
                out = out.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
                out = out.permute(0,2,1).contiguous()
                c_in = torch.repeat_interleave(c_list[i+1], dim=0, repeats=sc**(i+1))
        z = out

        log_p_sum = 0.5 * (- log(2.0 * pi) - z.pow(2)).sum()
        log_det = log_det_sum / (Bx * Cx * Tx)
        log_p = log_p_sum / (Bx * Cx * Tx)

        return log_p, log_det

    def reverse(self, z, mel):
        sc = self.sqz_scale

        c_list = self.upsample_conv(mel)
        out = self.sqz_layer(z)

        for i in range(len(self.ER_blocks)-1):
            B, C, T = out.shape
            out = out.permute(0,2,1).contiguous().view(B, (C*T)//sc, sc)
            out = out.permute(0,2,1).contiguous().view(B*sc, T//sc, C)
            out = out.permute(0,2,1).contiguous().view(B*sc, C, T//sc)

        c_in = torch.repeat_interleave(c_list[0], dim=0, repeats=sc**(len(self.ER_blocks)-1))

        for i, block in enumerate(self.ER_blocks[::-1]):
            out, _ = block.reverse(out, c_in)

            if i != len(self.ER_blocks)-1 :
                B, C, T = out.shape
                out = out.permute(0,2,1).contiguous()
                out = out.view(B//sc, sc, T, C).permute(0,2,3,1).contiguous()
                out = out.view(B//sc, T*sc, C).permute(0,2,1).contiguous()
                c_in = torch.repeat_interleave(c_list[i+1], dim=0, repeats=sc**(len(self.ER_blocks)-2-i))
        x = self.sqz_layer.reverse(out)

        return x