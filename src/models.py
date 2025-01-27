from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Parameter
from torch.autograd import Variable
import torch.utils.data
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn.init as init

import math



class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.LeakyReLU):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class RandomFourierEmbedding2d(nn.Module):
    def __init__(self, in_dim, embedding_dim, sigma=0.01):
        super(RandomFourierEmbedding2d, self).__init__()

        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

        w = torch.randn(embedding_dim, in_dim) #/ self.sigma  shape of weights: (out_features, in_features)
        b = torch.rand(embedding_dim)*2*np.pi

        self.register_buffer('weight', w)
        self.register_buffer('bias', b)

        print('# sigma value is {}'.format(self.sigma))


    def forward(self, x):
        if x is None:
            return 0

        z = torch.cos(F.linear(x, self.weight/self.sigma, self.bias))
        return z






class SpatialGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_out=1, num_layers=1, activation=nn.LeakyReLU
                , resid=False, fourier_expansion=False, sigma=0.01):
        super(SpatialGenerator, self).__init__()

        self.fourier_expansion = fourier_expansion

        in_dim = 2
        if fourier_expansion:
            embedding_dim = 1024
            self.embed_latent = RandomFourierEmbedding2d(in_dim, embedding_dim, sigma)
            in_dim = embedding_dim


        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1,num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):
        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)

        x = x.view(b*n, -1)

        if self.fourier_expansion:
            x = self.embed_latent(x)


        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)

        h_z = 0
        if hasattr(self, 'latent_linear'):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)

        h = h_x + h_z # (batch, num_coords, hidden_dim)
        h = h.view(b*n, -1)

        y = self.layers(h) # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        return y

class GroupConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_rot_dim=1, output_rot_dim=4):
        super(GroupConv, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        if type(padding) == int:
            padding = _pair(padding)
        else:
            padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_rot_dim = input_rot_dim
        self.output_rot_dim = output_rot_dim

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_rot_dim, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def trans_filter(self, device):
        '''
        Building r rotated filters
        '''
        res = torch.zeros(self.weight.shape[0], self.output_rot_dim,
                          self.weight.shape[1], self.weight.shape[2],
                          self.weight.shape[3], self.weight.shape[4]).to(device)
        d_theta = 2*np.pi / self.output_rot_dim
        theta = 0.0

        for i in range(self.output_rot_dim):
            #create the rotation matrix
            rot = torch.zeros(self.weight.shape[0], 3, 4).to(device)
            rot[:,0,0] = np.cos(theta)
            rot[:,0,1] = np.sin(theta)
            rot[:,1,0] = -np.sin(theta)
            rot[:,1,1] = np.cos(theta)

            grid = F.affine_grid(rot, self.weight.shape, align_corners=False)
            res[:, i, :, :, :] = F.grid_sample(self.weight, grid, align_corners=False)

            theta += d_theta

        return res

    def forward(self, input, device):
        tw = self.trans_filter(device)

        tw_shape = (self.out_channels*self.output_rot_dim,
                    self.in_channels*self.input_rot_dim,
                    self.ksize, self.ksize)

        tw = tw.view(tw_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_rot_dim, input_shape[-2],
                           input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)

        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_rot_dim, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class GroupConvSep(nn.Module):
    '''
    Depthwise-separable implementation of group convolution
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_rot_dim=1, output_rot_dim=4):
        super(GroupConvSep, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        if type(padding) == int:
            padding = _pair(padding)
        else:
            padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_rot_dim = input_rot_dim
        self.output_rot_dim = output_rot_dim

        self.weight1 = Parameter(torch.Tensor(
            1, in_channels, self.input_rot_dim, *kernel_size), requires_grad=True)
        self.weight2 = Parameter(torch.Tensor(
            self.output_rot_dim, out_channels, in_channels, self.output_rot_dim, self.input_rot_dim, 1, 1), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def trans_filter(self, device):
        '''
        Building r rotated filters
        '''
        res = torch.zeros(self.weight1.shape[0], self.output_rot_dim,
                          self.weight1.shape[1], self.weight1.shape[2],
                          self.weight1.shape[3], self.weight1.shape[4]).to(device)
        d_theta = 2*np.pi / self.output_rot_dim
        theta = 0.0

        for i in range(self.output_rot_dim):
            #create the rotation matrix
            rot = torch.zeros(self.weight1.shape[0], 3, 4).to(device)
            rot[:,0,0] = np.cos(theta)
            rot[:,0,1] = np.sin(theta)
            rot[:,1,0] = -np.sin(theta)
            rot[:,1,1] = np.cos(theta)

            grid = F.affine_grid(rot, self.weight1.shape, align_corners=False)
            res[:, i, :, :, :] = F.grid_sample(self.weight1, grid, align_corners=False)

            theta += d_theta

        return res

    def forward(self, input, device):
        tw1 = self.trans_filter(device)

        tw1_shape = (self.in_channels*self.input_rot_dim*self.output_rot_dim,
                    1,
                    self.ksize, self.ksize)
        tw2_shape = (self.out_channels*self.output_rot_dim,
                     self.in_channels*self.input_rot_dim*self.output_rot_dim,
                     1, 1)

        tw1 = tw1.view(tw1_shape)
        tw2 = self.weight2.view(tw2_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_rot_dim, input_shape[-2],
                           input_shape[-1])
        y1 = F.conv2d(input, weight=tw1, bias=None, stride=self.stride,
                        padding=self.padding, groups=self.in_channels*self.input_rot_dim)
        y2 = F.conv2d(y1, weight=tw2, bias=None)

        batch_size, _, ny_out, nx_out = y2.size()
        y2 = y2.view(batch_size, self.out_channels, self.output_rot_dim, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y2 = y2 + bias

        return y2

class GroupRotationWiseConv(GroupConv):
    '''
    In the forward function, the covolutional filters are only applied on the 
    part of the feature map with corresponding rotations.
    
    input_rot_dim is set to 1 since each rotation group only is applied on the 
    corresponding rotation group of the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_rot_dim=1, output_rot_dim=16):
        
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding, bias, input_rot_dim, output_rot_dim)
        
    
        
    def forward(self, input, device):
        # shape of input is (B, C_in, P_in, H, W)
        
        # rotates the self.weight as many as P_out times
        tw = self.trans_filter(device) # (C_out, P_out, C_in, 1, H, W); here P_in=1
        tw = tw.squeeze(3) # (C_out, P_out, C_in, H, W)
        
        y_groups = []
        for i in range(input.shape[2]):
            input_group = input[:, :, i, :, :] # (B, C_in, H, W); inputs coming from the ith P_in 
            tw_group = tw[:, i, :, :, :] # (C_out, C_in, H, W);  self.weights from the ith P_out
            y_groups.append(F.conv2d(input_group, weight=tw_group, bias=None, stride=self.stride,
                               padding=self.padding)) # (B, C_out, C_in, H, W)
        
        y = torch.stack(y_groups, dim=2).to(device) # (B, C_out, P_out, C_in, H, W)
        
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y

class DepthwiseRotationConv(nn.Module):
    '''
    1x1 conv to interpolate between rotation groups
    '''
    def __init__(self, in_channels, out_channels,
                 bias=True, input_rot_dim=4, output_rot_dim=4):

        super(DepthwiseRotationConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_rot_dim = input_rot_dim
        self.output_rot_dim = output_rot_dim

        kernel_size = (input_rot_dim, 1, 1)
        if input_rot_dim == output_rot_dim:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
            self.pad_dim = (0, 0, 0, 0, 0, input_rot_dim-1)
        elif input_rot_dim > output_rot_dim:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=(2,1,1), bias=False)
            self.pad_dim = (0, 0, 0, 0, (input_rot_dim-2)//2, (input_rot_dim-2)//2)
        else:
            kernel_size = (output_rot_dim, 1, 1)
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=(2,1,1), padding=(output_rot_dim-1, 0, 0), bias=False)
            pad_size = output_rot_dim//4
            self.pad_dim = (0, 0, 0, 0, pad_size, pad_size)
        self.kernel_size = kernel_size

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        torch.nn.init.uniform_(self.conv.weight, -stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, device):
 
        inp_pad = F.pad(input, self.pad_dim, 'circular')
        y = self.conv(inp_pad)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class InferenceNetwork_UnimodalTranslation_UnimodalRotation(nn.Module):
    '''
    Inference without attention on the translation and rotation values
    '''
    def __init__(self, n, latent_dim, hidden_dim, num_layers=1, activation=nn.LeakyReLU, resid=False):
        super(InferenceNetwork_UnimodalTranslation_UnimodalRotation, self).__init__()

        self.latent_dim = latent_dim
        self.n = n
        print('n is {}'.format(n))
        layers = [nn.Linear(n, hidden_dim),
                  activation(),
                 ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(ResidLinear(hidden_dim, hidden_dim, activation=activation))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 2*latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        z = self.layers(x)

        ld = self.latent_dim
        z_mu = z[:,:ld]
        z_logstd = z[:,ld:]

        return z_mu, z_logstd







class InferenceNetwork_AttentionTranslation_UnimodalRotation(nn.Module):
    '''
    Inference with attention only on the translation values
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, activation=nn.LeakyReLU, groupconv=0):

        super(InferenceNetwork_AttentionTranslation_UnimodalRotation, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.groupconv = groupconv

        if self.groupconv == 0:
            self.conv1 = nn.Conv2d(in_channels, self.kernels_num, self.input_size, padding=self.input_size//2)
            self.conv2 = nn.Conv2d(self.kernels_num, self.kernels_num, 1)

            self.conv_a = nn.Conv2d(self.kernels_num, 1, 1)
            self.conv_r = nn.Conv2d(self.kernels_num, 2, 1)
            self.conv_z = nn.Conv2d(self.kernels_num, 2*self.latent_dim, 1)
        else:
            self.conv1 = GroupConv(in_channels, self.kernels_num, self.input_size, padding=self.input_size//2, input_rot_dim=1, output_rot_dim=self.groupconv)
            self.conv2 = nn.Conv2d(self.kernels_num, self.kernels_num, 1)
            self.fc_r = nn.Linear(self.groupconv, 1)

            self.conv_a = nn.Conv2d(self.kernels_num, 1, 1)
            self.conv_r = nn.Conv2d(self.kernels_num, 2, 1)
            self.conv_z = nn.Conv2d(self.kernels_num, 2*self.latent_dim, 1)

    def forward(self, x, device):
        if self.groupconv > 0:
            x = self.activation(self.conv1(x, device))
            x = x.permute(0, 1, 3, 4, 2)
            x = self.fc_r(x).squeeze(4)
        else:
            x = self.activation(self.conv1(x))

        h = self.activation(self.conv2(x))

        attn = self.conv_a(h)
        a = attn.view(attn.shape[0], -1)
        a_sampled = F.gumbel_softmax(a, dim=-1)
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3])

        z = self.conv_z(h)

        theta = self.conv_r(h)

        return attn, a_sampled, theta, z



class InferenceNetwork_AttentionTranslation_AttentionRotation(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_size=65, padding=16, activation=nn.LeakyReLU
                 , groupconv=0, rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True):

        super(InferenceNetwork_AttentionTranslation_AttentionRotation, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_size = kernels_size
        self.padding = padding
        self.groupconv = groupconv
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r

        self.conv1 = GroupConv(in_channels, self.kernels_num, self.kernels_size, padding=self.padding, input_rot_dim=1, output_rot_dim=self.groupconv)
        self.conv2 = nn.Conv3d(self.kernels_num, self.kernels_num, 1)

        self.conv_a = nn.Conv3d(self.kernels_num, 1, 1)
        self.conv_r = nn.Conv3d(self.kernels_num, 2, 1)
        self.conv_z = nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1)


    def forward(self, x, device):
        x = self.activation(self.conv1(x, device))
        h = self.activation(self.conv2(x))

        attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)

        if self.rot_refinement:
            if self.groupconv == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float)
            elif self.groupconv == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float)
            elif self.groupconv == 16:
                offsets = torch.tensor([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi, -7*np.pi/8, -3*np.pi/4, -5*np.pi/8, -np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8]).type(torch.float)

            if self.normal_prior_over_r:
                prior_theta = Normal(torch.tensor([0.0]).to(device), torch.tensor([self.theta_prior]).to(device))
            else:
                prior_theta = Uniform(torch.tensor([-2*np.pi]).to(device), torch.tensor([2*np.pi]).to(device))

            offsets = offsets.to(device)
            p_r = prior_theta.log_prob(offsets).unsqueeze(1).unsqueeze(2)

        else:
            # uniform prior over r when no offsets are being added to the rot_means
            p_r = torch.zeros(self.groupconv).to(device) - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)

        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        a = attn.view(attn.shape[0], -1)

        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        z = self.conv_z(h)

        theta = self.conv_r(h)

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z


class InferenceNetwork_AttentionTranslation_AttentionRotationSep(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    This version of the network is modified to use separable convolutions.
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_size=65, padding=16, activation=nn.LeakyReLU
                 , groupconv=0, rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True):

        super(InferenceNetwork_AttentionTranslation_AttentionRotationSep, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_size = kernels_size
        self.padding = padding
        self.groupconv = groupconv
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r

        self.conv1 = GroupConvSep(in_channels, self.kernels_num, self.kernels_size, padding=self.padding, input_rot_dim=1, output_rot_dim=self.groupconv)
        self.conv2 = nn.Conv3d(self.kernels_num, self.kernels_num, 1)

        self.conv_a = nn.Conv3d(self.kernels_num, 1, 1)
        self.conv_r = nn.Conv3d(self.kernels_num, 2, 1)
        self.conv_z = nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1)


    def forward(self, x, device):
        x = self.activation(self.conv1(x, device))
        h = self.activation(self.conv2(x))

        attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)

        if self.rot_refinement:
            if self.groupconv == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float)
            elif self.groupconv == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float)
            elif self.groupconv == 16:
                offsets = torch.tensor([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi, -7*np.pi/8, -3*np.pi/4, -5*np.pi/8, -np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8]).type(torch.float)

            if self.normal_prior_over_r:
                prior_theta = Normal(torch.tensor([0.0]).to(device), torch.tensor([self.theta_prior]).to(device))
            else:
                prior_theta = Uniform(torch.tensor([-2*np.pi]).to(device), torch.tensor([2*np.pi]).to(device))

            offsets = offsets.to(device)
            p_r = prior_theta.log_prob(offsets).unsqueeze(1).unsqueeze(2)

        else:
            # uniform prior over r when no offsets are being added to the rot_means
            p_r = torch.zeros(self.groupconv).to(device) - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)

        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        a = attn.view(attn.shape[0], -1)

        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        z = self.conv_z(h)

        theta = self.conv_r(h)

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z

class InferenceNetwork_AttTra_AttRot_Searched(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_sizes=[65, 65], padding=16, activation=nn.LeakyReLU
                 , groupconvs=[0, 0], rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True):

        super(InferenceNetwork_AttTra_AttRot_Searched, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_sizes = kernels_sizes
        self.padding = padding
        self.groupconv = groupconvs[-1]
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r

        self.conv1 = GroupConv(in_channels, self.kernels_num, self.kernels_sizes[0], padding=self.padding, input_rot_dim=1, output_rot_dim=groupconvs[0])
        self.conv2 = GroupConv(self.kernels_num, self.kernels_num, self.kernels_sizes[1], padding="same", input_rot_dim=groupconvs[0], output_rot_dim=groupconvs[1])

        self.conv_a = nn.Conv3d(self.kernels_num, 1, 1)
        self.conv_r = nn.Conv3d(self.kernels_num, 2, 1)
        self.conv_z = nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1)


    def forward(self, x, device):
        x = self.activation(self.conv1(x, device))
        h = self.activation(self.conv2(x, device))

        attn = self.conv_a(h).squeeze(1) # <- 3dconv means this is (BxRxHxW)

        if self.rot_refinement:
            if self.groupconv == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float)
            elif self.groupconv == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float)
            elif self.groupconv == 16:
                offsets = torch.tensor([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi, -7*np.pi/8, -3*np.pi/4, -5*np.pi/8, -np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8]).type(torch.float)

            if self.normal_prior_over_r:
                prior_theta = Normal(torch.tensor([0.0]).to(device), torch.tensor([self.theta_prior]).to(device))
            else:
                prior_theta = Uniform(torch.tensor([-2*np.pi]).to(device), torch.tensor([2*np.pi]).to(device))

            offsets = offsets.to(device)
            p_r = prior_theta.log_prob(offsets).unsqueeze(1).unsqueeze(2)

        else:
            # uniform prior over r when no offsets are being added to the rot_means
            p_r = torch.zeros(self.groupconv).to(device) - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)

        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        a = attn.view(attn.shape[0], -1)

        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        z = self.conv_z(h)

        theta = self.conv_r(h)

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z

