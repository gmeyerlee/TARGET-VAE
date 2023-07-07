'''
Supernet version of TARGET-VAE code
'''

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
from models import *

class SuperGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_rot_dim=1, output_rot_dim=4):
        super(SuperGroupConv, self).__init__()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

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

class InferenceSuperNetwork_AttTra_AttRot(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_sizes=[65,], padding=16, activation=nn.LeakyReLU
                 , groupconvs=[0,], rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True, tau_init=10):

        super(InferenceSuperNetwork_AttTra_AttRot, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_sizes = kernels_sizes
        spat_red = max(kernels_sizes) - 2*padding
        self.padding = padding
        self.groupconvs = groupconvs
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.tau = tau_init
        self.ksize_weight = Parameter(1e-3 * torch.randn(len(self.kernels_sizes),), requires_grad=True)
        self.rdim_weight = Parameter(1e-3 * torch.randn(len(self.groupconvs),), requires_grad=True)

        self.conv1 = [GroupConv(in_channels, self.kernels_num, k_size, padding=(k_size - spat_red)/2, input_rot_dim=1, output_rot_dim=max(self.groupconvs)) for k_size in self.kernels_sizes]
        self.conv2 = [nn.Conv3d(self.kernels_num, self.kernels_num, 1) for _ in len(self.groupconvs)]

        self.conv_a = [nn.Conv3d(self.kernels_num, 1, 1) for _ in len(self.groupconvs)]
        self.conv_r = [nn.Conv3d(self.kernels_num, 2, 1) for _ in len(self.groupconvs)]
        self.conv_z = [nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in len(self.groupconvs)]

    def get_params(self):
        xlist = []
        for c1 in self.conv1:
            xlist += list(c1.parameters())
        for c2 in self.conv2:
            xlist += list(c2.parameters())
        for ca in self.conv_a:
            xlist += list(ca.parameters())
        for cr in self.conv_r:
            xlist += list(cr.parameters())
        for cz in self.conv_z:
            xlist += list(cz.parameters())
        return xlist

    def arch_params(self):
        return [self.ksize_weight, self.rdim_weight]

    def forward(self, x, device):

        alphas = F.gumbel_softmax(self.ksize_weight, tau=self.tau)
        x_out = sum([self.activation(conv_op(x, device)) * alpha for conv_op, alpha in zip(self.conv1, alphas)])
        rdim_max = max(self.groupconvs)
        xs_out = [x_out[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs]
        hs = [self.activation(c2(x_o)) for c2, x_o in zip(self.conv2, xs_out)]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        betas = F.gumbel_softmax(self.rdim_weight, tau=self.tau)
        attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[i], dim=1) * betas[i] for i, att in enumerate(attns)]) # <- 3dconv means this is (BxRxHxW)

        if self.rot_refinement:
            if max(self.groupconvs) == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float)
            elif max(self.groupconvs) == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float)
            elif max(self.groupconvs) == 16:
                offsets = torch.tensor([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi, -7*np.pi/8, -3*np.pi/4, -5*np.pi/8, -np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8]).type(torch.float)

            if self.normal_prior_over_r:
                prior_theta = Normal(torch.tensor([0.0]).to(device), torch.tensor([self.theta_prior]).to(device))
            else:
                prior_theta = Uniform(torch.tensor([-2*np.pi]).to(device), torch.tensor([2*np.pi]).to(device))

            offsets = offsets.to(device)
            p_r = prior_theta.log_prob(offsets).unsqueeze(1).unsqueeze(2)

        else:
            # uniform prior over r when no offsets are being added to the rot_means
            p_r = torch.zeros(rdim_max).to(device) - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)

        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        a = attn.view(attn.shape[0], -1)

        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[i], dim=2) * betas[i] for i, zi in enumerate(zs)])

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[i], dim=2) * betas[i] for i, th in enumerate(thetas)])

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z

class InferenceSuperNetwork_AttTra_AttRot2(nn.Module):
    '''
    Inference with attention on both the translation and rotation values (inference model for TARGET-VAE)
    This version has two sequential group convolutions
    '''
    def __init__(self, n, in_channels, latent_dim, kernels_num=128, kernels_sizes=[[65],[65]], padding=[16, 16], activation=nn.LeakyReLU
                 , groupconvs=[[0],[0]], rot_refinement=False, theta_prior=np.pi, normal_prior_over_r=True, tau_init=10):

        super(InferenceSuperNetwork_AttTra_AttRot2, self).__init__()

        self.activation = activation()
        self.latent_dim = latent_dim
        self.input_size = n
        self.kernels_num = kernels_num
        self.kernels_sizes = kernels_sizes
        spat_red1 = max(kernels_sizes[0]) - 2*padding[0]
        spat_red2 = max(kernels_sizes[1]) - 2*padding[1]
        self.padding = padding
        self.groupconvs = groupconvs
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.tau = tau_init
        self.ksize_weight = Parameter(1e-3 * torch.randn(len(self.kernels_sizes[0]), 2), requires_grad=True)
        self.rdim_weight = Parameter(1e-3 * torch.randn(len(self.groupconvs[0]), 2), requires_grad=True)

        self.conv1 = [GroupConv(in_channels, self.kernels_num, k_size, padding=(k_size - spat_red1)/2, input_rot_dim=1, output_rot_dim=max(self.groupconvs[0])) for k_size in self.kernels_sizes[0]]
        self.conv2 = [GroupConv(self.kernels_num, self.kernels_num, k_size, padding=(k_size - spat_red2)/2, input_rot_dim=max(self.groupconvs[0]), output_rot_dim=max(self.groupconvs[1])) for k_size in self.kernels_sizes[1]]

        self.conv_a = [nn.Conv3d(self.kernels_num, 1, 1) for _ in len(self.groupconvs)]
        self.conv_r = [nn.Conv3d(self.kernels_num, 2, 1) for _ in len(self.groupconvs)]
        self.conv_z = [nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in len(self.groupconvs)]

    def get_params(self):
        xlist = []
        for c1 in self.conv1:
            xlist += list(c1.parameters())
        for c2 in self.conv2:
            xlist += list(c2.parameters())
        for ca in self.conv_a:
            xlist += list(ca.parameters())
        for cr in self.conv_r:
            xlist += list(cr.parameters())
        for cz in self.conv_z:
            xlist += list(cz.parameters())
        return xlist

    def arch_params(self):
        return [self.ksize_weight, self.rdim_weight]

    def forward(self, x, device):

        alphas1 = F.gumbel_softmax(self.ksize_weight[:, 0], tau=self.tau)
        x_out = sum([self.activation(conv_op(x, device)) * alpha for conv_op, alpha in zip(self.conv1, alphas1)])
        xs_out = [x_out[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs[0]]
        betas1 = F.gumbel_softmax(self.rdim_weight[:, 0], tau=self.tau)
        x_out = sum([torch.repeat_interleave(x_o, max(self.groupconvs[0])//self.groupconvs[0][i], dim=2) * betas1[i] for i, x_o in enumerate(xs_out)])
        alphas2 = F.gumbel_softmax(self.ksize_weight[:, 1], tau=self.tau)
        h = sum([self.activation(conv_op(x_out, device)) * alpha for conv_op, alpha in zip(self.conv2, alphas2)])
        rdim_max = max(self.groupconvs[1])
        hs = [h[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs[1]]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        betas2 = F.gumbel_softmax(self.rdim_weight[:, 1], tau=self.tau)
        attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[1][i], dim=1) * betas2[i] for i, att in enumerate(attns)]) # <- 3dconv means this is (BxRxHxW)

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
            p_r = torch.zeros(rdim_max).to(device) - np.log(attn.shape[1])
            p_r = p_r.unsqueeze(1).unsqueeze(2)

        attn = attn + p_r
        q_t_r = F.log_softmax(attn.view(attn.shape[0], -1), dim=1).view(attn.shape[0], attn.shape[1], attn.shape[2], attn.shape[3]) # B x R x H x W

        a = attn.view(attn.shape[0], -1)

        a_sampled = F.gumbel_softmax(a, dim=-1) #
        a_sampled = a_sampled.view(h.shape[0], h.shape[2], h.shape[3], h.shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[1][i], dim=2) * betas2[i] for i, zi in enumerate(zs)])

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[1][i], dim=2) * betas2[i] for i, th in enumerate(thetas)])

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z


