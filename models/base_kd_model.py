from collections import OrderedDict

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class BaseKD(SRModel):
    """Base logits-based KD model."""

    def __init__(self, opt):
        super(BaseKD, self).__init__(opt)
        # Build Teacher Model
        self.net_t = build_network(opt['network_t'])
        self.net_t = self.model_to_device(self.net_t)
        # Load Teacher Model
        load_path_t = self.opt['path'].get('pretrain_network_t', None)
        assert load_path_t is not None, "checkpoint of the teacher network should be specified in opt.path.pretrain_network_t"
        if load_path_t is not None:
            param_key = self.opt['path'].get('param_key_t', 'params')
            self.load_network(self.net_t, load_path_t, True, param_key)

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        # build KD loss
        if train_opt.get('kd_opt'):
            self.cri_kd = build_loss(train_opt['kd_opt']).to(self.device)
        else:
            self.cri_kd = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_kd is None:
            raise ValueError('All the pixel, KD and perceptual losses are None.')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        self.output_t = self.net_t(self.lq)

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

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
