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
from .models import *

class SuperDoubleConv(nn.Module):
    '''
    NAS search cell for a sequential pair of Rotationwise GroupConvs
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #self.alphas1 = alphas1
        #self.alphas2 = alphas2
        #self.betas = betas
        self.tau = 1
            
        self.layer1 = nn.ModuleList([GroupRotationWiseConv(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
                       GroupRotationWiseConv(in_channels, in_channels, 5, stride=1, padding=2, bias=False),
                       GroupRotationWiseConv(in_channels, in_channels, 7, stride=1, padding=3, bias=False),
                       GroupRotationWiseConv(in_channels, 2*in_channels, 3, stride=1, padding=1, bias=False),
                       GroupRotationWiseConv(in_channels, 2*in_channels, 5, stride=1, padding=2, bias=False),
                       GroupRotationWiseConv(in_channels, 2*in_channels, 7, stride=1, padding=3, bias=False)])
        self.layer2 = nn.ModuleList([GroupRotationWiseConv(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                       GroupRotationWiseConv(in_channels, out_channels, 5, stride=stride, padding=2, bias=False),
                       GroupRotationWiseConv(in_channels, out_channels, 7, stride=stride, padding=3, bias=False),
                       GroupRotationWiseConv(in_channels, 2*out_channels, 3, stride=stride, padding=1, bias=False),
                       GroupRotationWiseConv(in_channels, 2*out_channels, 5, stride=stride, padding=2, bias=False),
                       GroupRotationWiseConv(in_channels, 2*out_channels, 7, stride=stride, padding=3, bias=False)])
        self.ri_conv = nn.ModuleList([DepthwiseRotationConv(out_channels, out_channels, input_rot_dim=4, output_rot_dim=4),
                        DepthwiseRotationConv(2*in_channels, in_channels, input_rot_dim=4, output_rot_dim=8),
                        DepthwiseRotationConv(2*out_channels, out_channels, input_rot_dim=4, output_rot_dim=8),
                        DepthwiseRotationConv(out_channels, out_channels, input_rot_dim=8, output_rot_dim=4),
                        DepthwiseRotationConv(out_channels, out_channels, input_rot_dim=8, output_rot_dim=8),
                        DepthwiseRotationConv(2*out_channels, out_channels, input_rot_dim=8, output_rot_dim=16),
                        DepthwiseRotationConv(in_channels, in_channels, input_rot_dim=16, output_rot_dim=8),
                        DepthwiseRotationConv(out_channels, out_channels, input_rot_dim=16, output_rot_dim=8),
                        DepthwiseRotationConv(out_channels, out_channels, input_rot_dim=16, output_rot_dim=16)])

        self.activation = nn.LeakyReLU(inplace=True)

    def get_params(self):
        xlist = []
        for c1 in self.layer1:
            xlist += list(c1.parameters())
        for c2 in self.layer2:
            xlist += list(c2.parameters())
        for ci in self.ri_conv:
            xlist += list(ci.parameters())
        return xlist

    def forward(self, x, alphas1, alphas2, betas, device):
        alp1 = F.gumbel_softmax(alphas1, tau=self.tau)
        alp2 = F.gumbel_softmax(alphas2, tau=self.tau)
        beta = F.gumbel_softmax(betas, tau=self.tau)

        stem1 = self.activation(sum([a1 * gconv(x, device) for a1, gconv in zip(alp1, self.layer1[:3])]))
        stem1up = self.activation(sum([a1 * gconv(x[:, :, ::4, :, :], device) for a1, gconv in zip(alp1, self.layer1[3:])]))
        stem2 = self.activation(sum([a2 * gconv(stem1, device) for a2, gconv in zip(alp2, self.layer2[:3])]))
        stem2up = self.activation(sum([a2 * gconv(stem1[:, :, ::2, :, :], device) for a2, gconv in zip(alp2, self.layer2[3:])]))
        out = beta[2]*stem2
       	out[:, :, ::2, :, :] += beta[1]*stem2[:, :, ::2, :, :]
        out[:, :, ::4, :, :] += beta[0]*stem2[:, :, ::4, :, :]
        out[:, :, ::4, :, :] += beta[3]*self.activation(self.ri_conv[0](stem2[:, :, ::4, :, :], device))
        out[:, :, ::2, :, :] += beta[4]*self.activation(self.ri_conv[2](stem2up[:, :, ::2, :, :], device))
        up_inter = self.activation(self.ri_conv[1](stem1up, device))
        up_inter = self.activation(sum([a2 * gconv(up_inter, device) for a2, gconv in zip(alp2, self.layer2[3:])]))
        out += beta[5]*self.activation(self.ri_conv[5](up_inter, device))
        out[:, :, ::4, :, :] += beta[6]*self.activation(self.ri_conv[3](stem2[:, :, ::2, :, :], device))
        out[:, :, ::2, :, :] += beta[7]*self.activation(self.ri_conv[4](stem2[:, :, ::2, :, :], device))
        out += beta[8]*self.activation(self.ri_conv[5](stem2up, device))
        down_inter = self.activation(self.ri_conv[6](stem1, device))
        down_inter = self.activation(sum([a2 * gconv(down_inter, device) for a2, gconv in zip(alp2, self.layer2[:3])]))
        out[:, :, ::4, :, :] += beta[9]*self.activation(self.ri_conv[3](down_inter, device))
        out[:, :, ::2, :, :] += beta[10]*self.activation(self.ri_conv[7](stem2, device))
        out += beta[11]*self.activation(self.ri_conv[8](stem2, device))

        return out

class SuperDownNetMNIST(nn.Module):
    '''
    Inference Network with several layers of downsampling groupconv
    '''
    def __init__(self, rot_refinement=True, in_channels=1, latent_dim=2, theta_prior=np.pi, normal_prior_over_r=True, tau_init=10):
        super(SuperDownNetMNIST, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.tau = tau_init
        self.groupconvs = [4, 8, 16]
        self.rot_refinement=rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.kernels_num=64


        self.alphas0 = Parameter(1e-3 * torch.randn(3,), requires_grad=True)
        self.alphas1 = Parameter(1e-3 * torch.randn(3,2), requires_grad=True) 
        self.alphas2 = Parameter(1e-3 * torch.randn(3,2), requires_grad=True)
        self.betas = Parameter(1e-3 * torch.randn(12,2), requires_grad=True)
        self.betas_out = Parameter(1e-3 * torch.randn(3,), requires_grad=True)

        self.inc = nn.ModuleList([GroupConv(in_channels, 16, 3, padding=1, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 16, 5, padding=2, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 16, 7, padding=3, input_rot_dim=1, output_rot_dim=16)])
        #self.down1 = SuperDoubleConv(16, 32, self.alphas1[:,0], self.alphas2[:,0], self.betas[:,0], stride=2)
        #self.down2 = SuperDoubleConv(32, 64, self.alphas1[:,1], self.alphas2[:,1], self.betas[:,1], stride=2)
        self.down1 = SuperDoubleConv(16, 32, stride=2)
        self.down2 = SuperDoubleConv(32, 64, stride=2)

 
        self.conv_a = nn.ModuleList([nn.Conv3d(self.kernels_num, 1, 1) for _ in self.groupconvs])
        self.conv_r = nn.ModuleList([nn.Conv3d(self.kernels_num, 2, 1) for _ in self.groupconvs])
        self.conv_z = nn.ModuleList([nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in self.groupconvs])

    def get_params(self):
        xlist = []
        xlist += self.down1.get_params()
        xlist += self.down2.get_params()
        for ca in self.conv_a:
            xlist += list(ca.parameters())
        for cr in self.conv_r:
            xlist += list(cr.parameters())
        for cz in self.conv_z:
            xlist += list(cz.parameters())
        return xlist

    def get_arch_params(self):
        return [self.alphas0, self.alphas1, self.alphas2,
                self.betas, self.betas_out]

    def set_tau(self, tau):
        self.tau = tau
        self.down1.tau = tau
        self.down2.tau = tau

    def forward(self, x, device):
        alp0 = F.gumbel_softmax(self.alphas0, tau=self.tau)
        b_o = F.gumbel_softmax(self.betas_out, tau=self.tau)
        x = sum([a0 * iconv(x, device) for a0, iconv in zip(alp0, self.inc)])
        x = self.down1(x, self.alphas1[:, 0], self.alphas2[:, 0], self.betas[:, 0], device)
        h = self.down2(x, self.alphas1[:, 1], self.alphas2[:, 1], self.betas[:, 1], device)

        rdim_max = max(self.groupconvs)
        hs = [h[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        for i, att in enumerate(attns):
            if torch.any(torch.isnan(att)).item():
                print(i, att)
        attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[i], dim=1) * b_o[i] for i, att in enumerate(attns)])

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
        a_sampled = a_sampled.view(hs[-1].shape[0], hs[-1].shape[2], hs[-1].shape[3], hs[-1].shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, zi in enumerate(zs)])

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, th in enumerate(thetas)])

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z

class SuperUNetMNIST(nn.Module):
    '''
    Inference Network with several layers of downsampling groupconv and corresponding upsampling
    '''
    def __init__(self, rot_refinement=True, in_channels=1, latent_dim=2, theta_prior=np.pi, normal_prior_over_r=True, tau_init=10):
        super(SuperUNetMNIST, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.tau = tau_init
        self.groupconvs = [4, 8, 16]
        self.rot_refinement=rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.kernels_num=16


        self.alphas0 = Parameter(1e-3 * torch.randn(3,), requires_grad=True)
        self.alphas1 = Parameter(1e-3 * torch.randn(3,4), requires_grad=True) 
        self.alphas2 = Parameter(1e-3 * torch.randn(3,4), requires_grad=True)
        self.betas = Parameter(1e-3 * torch.randn(12,4), requires_grad=True)
        self.betas_out = Parameter(1e-3 * torch.randn(3,), requires_grad=True)

        self.inc = nn.ModuleList([GroupConv(in_channels, 16, 3, padding=1, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 16, 5, padding=2, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 16, 7, padding=3, input_rot_dim=1, output_rot_dim=16)])
        self.down1 = SuperDoubleConv(16, 32, stride=2)
        self.down2 = SuperDoubleConv(32, 64, stride=2)
        self.upsamp1 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        self.upsamp2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)

        self.up1 = SuperDoubleConv(96, 32)
        self.up2 = SuperDoubleConv(48, 16)

 
        self.conv_a = nn.ModuleList([nn.Conv3d(self.kernels_num, 1, 1) for _ in self.groupconvs])
        self.conv_r = nn.ModuleList([nn.Conv3d(self.kernels_num, 2, 1) for _ in self.groupconvs])
        self.conv_z = nn.ModuleList([nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in self.groupconvs])

    def get_params(self):
        xlist = []
        xlist += self.down1.get_params()
        xlist += self.down2.get_params()
        xlist += self.up1.get_params()
        xlist += self.up2.get_params()

        for ca in self.conv_a:
            xlist += list(ca.parameters())
        for cr in self.conv_r:
            xlist += list(cr.parameters())
        for cz in self.conv_z:
            xlist += list(cz.parameters())
        return xlist

    def get_arch_params(self):
        return [self.alphas0, self.alphas1, self.alphas2,
                self.betas, self.betas_out]

    def set_tau(self, tau):
        self.tau = tau
        self.down1.tau = tau
        self.down2.tau = tau
        self.up1.tau = tau
        self.up2.tau = tau


    def forward(self, x, device):

        def pad_upsample(x, encFeatures):
            diffY = x.size()[3] - encFeatures.size()[3]
            diffX = x.size()[4] - encFeatures.size()[4]
            encFeatures = F.pad(encFeatures, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            return torch.cat([encFeatures, x], dim=1)

        alp0 = F.gumbel_softmax(self.alphas0, tau=self.tau)
        b_o = F.gumbel_softmax(self.betas_out, tau=self.tau)
        x = sum([a0 * iconv(x, device) for a0, iconv in zip(alp0, self.inc)])
        x1 = self.down1(x, self.alphas1[:, 0], self.alphas2[:, 0], self.betas[:, 0], device)
        x2 = self.down2(x1, self.alphas1[:, 1], self.alphas2[:, 1], self.betas[:, 1], device)
        x2_up = self.upsamp1(x2)
        x3 = self.up1(pad_upsample(x2_up, x1), self.alphas1[:, 2], self.alphas2[:, 2], self.betas[:, 2], device)
        x3_up = self.upsamp2(x3)
        h = self.up2(pad_upsample(x3_up, x), self.alphas1[:, 3], self.alphas2[:, 3], self.betas[:, 3], device)

        rdim_max = max(self.groupconvs)
        hs = [h[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[i], dim=1) * b_o[i] for i, att in enumerate(attns)])

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
        a_sampled = a_sampled.view(hs[-1].shape[0], hs[-1].shape[2], hs[-1].shape[3], hs[-1].shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, zi in enumerate(zs)])

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, th in enumerate(thetas)])

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z


class SuperDownNetEMPIAR(nn.Module):
    '''
    Inference Network with several layers of downsampling groupconv
    '''
    def __init__(self, rot_refinement=True, in_channels=1, latent_dim=2, theta_prior=np.pi, normal_prior_over_r=True, tau_init=10):
        super(SuperDownNetEMPIAR, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.tau = tau_init
        self.groupconvs = [4, 8, 16]
        self.rot_refinement=rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.kernels_num=128


        self.alphas0 = Parameter(1e-3 * torch.randn(3,), requires_grad=True)
        self.alphas1 = Parameter(1e-3 * torch.randn(3,4), requires_grad=True) 
        self.alphas2 = Parameter(1e-3 * torch.randn(3,4), requires_grad=True)
        self.betas = Parameter(1e-3 * torch.randn(12,4), requires_grad=True)
        self.betas_out = Parameter(1e-3 * torch.randn(3,), requires_grad=True)

        self.inc = nn.ModuleList([GroupConv(in_channels, 8, 3, padding=1, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 8, 5, padding=2, input_rot_dim=1, output_rot_dim=16),
                    GroupConv(in_channels, 8, 7, padding=3, input_rot_dim=1, output_rot_dim=16)])
        self.down1 = SuperDoubleConv(8, 16, stride=2)
        self.down2 = SuperDoubleConv(16, 32, stride=2)
        self.down3 = SuperDoubleConv(32, 64, stride=2)
        self.down4 = SuperDoubleConv(64, 128, stride=2)

 
        self.conv_a = nn.ModuleList([nn.Conv3d(self.kernels_num, 1, 1) for _ in self.groupconvs])
        self.conv_r = nn.ModuleList([nn.Conv3d(self.kernels_num, 2, 1) for _ in self.groupconvs])
        self.conv_z = nn.ModuleList([nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in self.groupconvs])

    def get_params(self):
        xlist = []
        xlist += self.down1.get_params()
        xlist += self.down2.get_params()
        xlist += self.down3.get_params()
        xlist += self.down4.get_params()
        for ca in self.conv_a:
            xlist += list(ca.parameters())
        for cr in self.conv_r:
            xlist += list(cr.parameters())
        for cz in self.conv_z:
            xlist += list(cz.parameters())
        return xlist

    def get_arch_params(self):
        return [self.alphas0, self.alphas1, self.alphas2,
                self.betas, self.betas_out]

    def set_tau(self, tau):
        self.tau = tau
        self.down1.tau = tau
        self.down2.tau = tau
        self.down3.tau = tau
        self.down4.tau = tau

    def forward(self, x, device):
        alp0 = F.gumbel_softmax(self.alphas0, tau=self.tau)
        b_o = F.gumbel_softmax(self.betas_out, tau=self.tau)
        x = sum([a0 * iconv(x, device) for a0, iconv in zip(alp0, self.inc)])
        x = self.down1(x, self.alphas1[:, 0], self.alphas2[:, 0], self.betas[:, 0], device)
        x = self.down2(x, self.alphas1[:, 1], self.alphas2[:, 1], self.betas[:, 1], device)
        x = self.down3(x, self.alphas1[:, 2], self.alphas2[:, 2], self.betas[:, 2], device)
        h = self.down4(x, self.alphas1[:, 3], self.alphas2[:, 3], self.betas[:, 3], device)

        rdim_max = max(self.groupconvs)
        hs = [h[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[i], dim=1) * b_o[i] for i, att in enumerate(attns)])

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
        a_sampled = a_sampled.view(hs[-1].shape[0], hs[-1].shape[2], hs[-1].shape[3], hs[-1].shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, zi in enumerate(zs)])

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[i], dim=2) * b_o[i] for i, th in enumerate(thetas)])

        if self.rot_refinement:
            rotation_offset = torch.ones_like(a_sampled) * offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            theta_mu = theta[ :, 0, :, :, : ] + rotation_offset
            theta_std = theta[ :, 1, :, :, : ]
            theta = torch.stack((theta_mu, theta_std), dim=1)
        else:
            offsets = torch.tensor([0]*attn.shape[1]).type(torch.float).to(device)
        return attn, q_t_r, p_r, a_sampled, offsets, theta, z


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

        self.conv1 = nn.ModuleList([GroupConv(in_channels, self.kernels_num, k_size, padding=(k_size - spat_red)//2, input_rot_dim=1, output_rot_dim=max(self.groupconvs)) for k_size in self.kernels_sizes])
        self.conv2 = nn.ModuleList([nn.Conv3d(self.kernels_num, self.kernels_num, 1) for _ in self.groupconvs])

        self.conv_a = nn.ModuleList([nn.Conv3d(self.kernels_num, 1, 1) for _ in self.groupconvs])
        self.conv_r = nn.ModuleList([nn.Conv3d(self.kernels_num, 2, 1) for _ in self.groupconvs])
        self.conv_z = nn.ModuleList([nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in self.groupconvs])

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

    def get_arch_params(self):
        return [self.ksize_weight, self.rdim_weight]

    def forward(self, x, device):

        def scatter_sum(op_outs, weights, dim=2):
            max_out = op_outs[-1]
            max_dim = max_out.shape[dim]
            assert max_dim > op_outs[0].shape[dim]
            to_sum = []
            for i, op_out in enumerate(op_outs[:-1]):
                expand_out = torch.zeros_like(max_out).slice_scatter(op_out, dim=dim, step=max_dim//op_out.shape[dim])
                to_sum.append(expand_out * weights[i])
            to_sum.append(max_out * weights[-1])
            return sum(to_sum)

        alphas = F.gumbel_softmax(self.ksize_weight, tau=self.tau)
        x_out = sum([self.activation(conv_op(x, device)) * alpha for conv_op, alpha in zip(self.conv1, alphas)])
        rdim_max = max(self.groupconvs)
        xs_out = [x_out[:, :, ::(rdim_max//rdim), :, :] for rdim in self.groupconvs]
        hs = [self.activation(c2(x_o)) for c2, x_o in zip(self.conv2, xs_out)]

        attns = [c_a(h).squeeze(1) for c_a, h in zip(self.conv_a, hs)]
        betas = F.gumbel_softmax(self.rdim_weight, tau=self.tau)
        #attn = sum([torch.repeat_interleave(att, rdim_max//self.groupconvs[i], dim=1) * betas[i] for i, att in enumerate(attns)]) # <- 3dconv means this is (BxRxHxW)
        attn = scatter_sum(attns, betas, dim=1)

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
        a_sampled = a_sampled.view(hs[-1].shape[0], hs[-1].shape[2], hs[-1].shape[3], hs[-1].shape[4])

        zs = [c_z(h) for c_z, h in zip(self.conv_z, hs)]
        #z = sum([torch.repeat_interleave(zi, rdim_max//self.groupconvs[i], dim=2) * betas[i] for i, zi in enumerate(zs)])
        z = scatter_sum(zs, betas)

        thetas = [c_r(h) for c_r, h in zip(self.conv_z, hs)]
        #theta = sum([torch.repeat_interleave(th, rdim_max//self.groupconvs[i], dim=2) * betas[i] for i, th in enumerate(thetas)])
        theta = scatter_sum(thetas, betas)

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
        #spat_red2 = max(kernels_sizes[1]) - 2*padding[1]
        self.padding = padding
        self.groupconvs = groupconvs
        self.rot_refinement = rot_refinement
        self.theta_prior = theta_prior
        self.normal_prior_over_r = normal_prior_over_r
        self.tau = tau_init
        self.ksize_weight = Parameter(1e-3 * torch.randn(len(self.kernels_sizes[0]), 2), requires_grad=True)
        self.rdim_weight = Parameter(1e-3 * torch.randn(len(self.groupconvs[0]), 2), requires_grad=True)

        self.conv1 = nn.ModuleList([GroupConv(in_channels, self.kernels_num, k_size, padding=(k_size - spat_red1)//2, input_rot_dim=1, output_rot_dim=max(self.groupconvs[0])) for k_size in self.kernels_sizes[0]])
        self.conv2 = nn.ModuleList([GroupConv(self.kernels_num, self.kernels_num, k_size, padding='same', input_rot_dim=max(self.groupconvs[0]), output_rot_dim=max(self.groupconvs[1])) for k_size in self.kernels_sizes[1]])

        self.conv_a = nn.ModuleList([nn.Conv3d(self.kernels_num, 1, 1) for _ in self.groupconvs[1]])
        self.conv_r = nn.ModuleList([nn.Conv3d(self.kernels_num, 2, 1) for _ in self.groupconvs[1]])
        self.conv_z = nn.ModuleList([nn.Conv3d(self.kernels_num, 2*self.latent_dim, 1) for _ in self.groupconvs[1]])

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

    def get_arch_params(self):
        return [self.ksize_weight, self.rdim_weight]

    def forward(self, x, device):

        alphas1 = F.gumbel_softmax(self.ksize_weight[:, 0], tau=self.tau)
        x_out = sum([self.activation(conv_op(x, device)) * alpha for conv_op, alpha in zip(self.conv1, alphas1)])
        rdim_max = max(self.groupconvs[0])
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
            if max(self.groupconvs[1]) == 4:
                offsets = torch.tensor([0, np.pi/2, np.pi, -np.pi/2]).type(torch.float)
            elif max(self.groupconvs[1]) == 8:
                offsets = torch.tensor([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]).type(torch.float)
            elif max(self.groupconvs[1]) == 16:
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
        a_sampled = a_sampled.view(hs[-1].shape[0], hs[-1].shape[2], hs[-1].shape[3], hs[-1].shape[4])

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


