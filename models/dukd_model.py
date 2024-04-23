import torch
import numpy as np
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.matlab_functions import imresize

from .base_kd_model import BaseKD

def noising(imgs, idx, img_range=1.):
    # imgs: torch tensor BxCxHxW
    ops = ['i', 'h', 'v', 't']
    imgs[idx['i']] = img_range - imgs[idx['i']]
    imgs[idx['h']] = imgs[idx['h']].flip(dims=(-2,))
    imgs[idx['v']] = imgs[idx['v']].flip(dims=(-1,))
    imgs[idx['t']] = imgs[idx['t']].clone().transpose(-2, -1)
    return imgs

@MODEL_REGISTRY.register()
class DUKD(BaseKD):
    """Data Upcycling Knowledge Distillation"""

    def __init__(self, opt):
        super(DUKD, self).__init__(opt)

    def init_training_settings(self):
        self.zoom_out = self.opt['train']['dukd_opt'].get('zoom_out', False)
        self.n_gt_sub = self.opt['train']['dukd_opt'].get('n_gt_sub', [1])
        super().init_training_settings()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        self.output_t = self.net_t(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)  # reduction not performed
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_kd:
            l_kd = self.cri_kd(self.output, self.output_t)
            l_total += l_kd
            loss_dict['l_kd'] = l_kd

        l_total.backward()
        self.optimizer_g.step()
        
        l_total = 0
        lr_shape = self.lq.shape[-1]
        s = self.opt['scale']

        idxs = np.array([(i,j) for i in range(s) for j in range(s)])[np.random.permutation(s**2)[:self.n_gt_sub[0]]]
        gt_sub = torch.cat([self.gt[..., i*lr_shape:(i+1)*lr_shape, j*lr_shape:(j+1)*lr_shape] for i, j in idxs], 0)
        sr_gt_sub = self.net_t(gt_sub)
        
        if self.opt['train'].get('noisy', False):
            ops = ['i', 'h', 'v', 't']  # invert color, horizontal, vertical flip, transpose
            prob = self.opt['train']['noisy'].get('prob', 0.3)
            noising_idx = {op: torch.nonzero(torch.Tensor(np.random.choice([0, 1], size=gt_sub.shape[0],
                                                                           p=[1-prob, prob]))
                                            ).squeeze()
                            for op in ops}
            gt_sub = noising(gt_sub, noising_idx)
            sr_gt_sub_g = noising(self.net_g(gt_sub), noising_idx)
            l_zoom_in = self.cri_kd(sr_gt_sub, sr_gt_sub_g)
        else:
            l_zoom_in = self.cri_kd(sr_gt_sub, self.net_g(gt_sub))

        l_total += l_zoom_in

        if self.zoom_out:
            lqlq = torch.stack([imresize(lq, 1/self.opt['scale']).to(self.device) for lq in self.lq.cpu()])
            sr_lqlq = self.net_g(lqlq)
            l_zoom_out_pix = self.cri_pix(sr_lqlq, self.lq)
            l_zoom_out_kd = self.cri_kd(sr_lqlq, self.net_t(lqlq))
            del sr_lqlq
            loss_dict['l_zoom_out_pix'] = l_zoom_out_pix
            loss_dict['l_zoom_out_kd'] = l_zoom_out_kd
            l_total += l_zoom_out_pix + l_zoom_out_kd

        l_total.backward()
        self.optimizer_g.step()

        del gt_sub, sr_gt_sub

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)