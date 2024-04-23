from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.rcan_arch import RCAN

from .base_kd_model import BaseKD

def fitnet(fm):
    return fm.mean(axis=-3, keepdim=True)

def spatial_similarity(fm):
    # He, Z., Dai, T., Lu, J., Jiang, Y., & Xia, S.-T. (2020). Fakd: Feature-Affinity Based Knowledge Distillation
    # for Efficient Image Super-Resolution. 2020 IEEE International Conference on Image Processing (ICIP), 518–522.
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 1e-8)
    s = norm_fm.transpose(1, 2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm):
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 2)).unsqueeze(2).expand(fm.shape) + 1e-8)
    s = norm_fm.bmm(norm_fm.transpose(1, 2))
    s = s.unsqueeze(1)
    return s

def batch_similarity(fm):
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0, 1))
    normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q

def AT(fm):
    # Sergey Zagoruyko and Nikos Komodakis, “Paying more attention to attention: Improving the performance of
    # convolutional neural networks via attention transfer,” arXiv preprint arXiv:1612.03928, 2016.
    eps = 1e-6
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    return am

def FSP(fm1, fm2):
    # Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim, “A gift from knowledge distillation: Fast optimization,
    # network minimization and transfer learning,” in CVPR, 2017.
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)
    fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    return fsp


class EDSR(EDSR):
    def forward_feats(self, x, feats=[]):
        assert feats, "The position of feature maps should be provided."
        feature_maps = []
        
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        if 'conv_first' in feats:
            feature_maps.append(x)

        res = x
        for i, rb in enumerate(self.body):
            res = rb(res)
            if i in feats or str(i) in feats:
                feature_maps.append(res)

        res = self.conv_after_body(res)
        if 'conv_after_body' in feats:
            feature_maps.append(res)

        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x, feature_maps


class RCAN(RCAN):
    def forward_feats(self, x, feats=[]):
        assert feats, "The position of feature maps should be provided."
        feature_maps = []

        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        if 'conv_first' in feats:
            feature_maps.append(x)

        res = x
        for i, rg in enumerate(self.body):
            res = rg(res)
            if i in feats or str(i) in feats:
                feature_maps.append(res)

        res = self.conv_after_body(res)
        if 'conv_after_body' in feats:
            feature_maps.append(res)

        res += x

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x, feature_maps


def build_network(opt):
    opt = __import__('copy').deepcopy(opt)
    network_type = opt.pop('type')
    if network_type == 'EDSR':
        return EDSR(**opt)
    elif network_type == 'RCAN':
        return RCAN(**opt)
    else:
        raise Exception("The FAKD only supports EDSR and RCAN.")


@MODEL_REGISTRY.register()
class FAKD(BaseKD):
    # Reference: https://github.com/Vincent-Hoo/Knowledge-Distillation-for-Super-resolution
    allowed_transformations = {
        'SA': spatial_similarity,
        'Fitnet': fitnet,
        'AT': AT,
        'FSP': FSP,
        'CA': channel_similarity,
        'IA': batch_similarity
    }

    def __init__(self, opt):
        super(FAKD, self).__init__(opt)
        feature_opt = opt['train'].get('feature_distillation', None)
        assert feature_opt is not None, "Options of feature distillation should be specified in opt.train"
        positions = feature_opt['positions']
        if isinstance(positions, list):
            self.feature_t = self.feature_g = positions
        elif isinstance(positions, dict):
            self.feature_t = positions.get('t')
            self.feature_g = positions.get('g')
        self.distillation_type = feature_opt.get('type')
        self.fm_transform = self.allowed_transformations[self.distillation_type]
        assert len(self.feature_t) == len(self.feature_g), \
            "Number of selected features maps for student and teacher are different!"
    
        # Re-initialize net_t
        del self.net_t
        self.net_t = self.model_to_device(build_network(opt['network_t']))
        self.load_network(self.net_t, self.opt['path'].get('pretrain_network_t', None), True, self.opt['path'].get('param_key_t', 'params'))

    def init_training_settings(self):
        # Re-initialize net_g
        del self.net_g
        self.net_g = self.model_to_device(build_network(self.opt['network_g']))
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), self.opt['path'].get('param_key_g', 'params'))

        try:
            super().init_training_settings()
        except ValueError:
            pass

        train_opt = self.opt['train']
        if train_opt.get('feature_opt'):
            self.cri_feature = build_loss(train_opt['feature_opt']).to(self.device)
        else:
            self.cri_feature = None
        if self.cri_pix is None and self.cri_perceptual is None \
            and self.cri_kd is None and self.cri_feature is None:
            raise ValueError('All the pixel, KD, perceptual and feature losses are None.')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output, fms_g = self.get_bare_model(self.net_g).forward_feats(self.lq, feats=self.feature_g)
        self.output_t, fms_t = self.get_bare_model(self.net_t).forward_feats(self.lq, feats=self.feature_t)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_kd:
            l_kd = self.cri_kd(self.output, self.output_t)
            l_total += l_kd
            loss_dict['l_kd'] = l_kd

        agg_fms_g = [self.fm_transform(fm) for fm in fms_g]
        agg_fms_t = [self.fm_transform(fm) for fm in fms_t]

        l_feature = torch.sum(torch.tensor([self.cri_feature(afg, aft) for afg, aft in zip(agg_fms_g, agg_fms_t)]))

        l_total += l_feature
        loss_dict['l_feature'] = l_feature

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
