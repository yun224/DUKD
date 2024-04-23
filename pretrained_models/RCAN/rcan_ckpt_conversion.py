import os
import yaml

import torch
from torch.nn import Parameter
from basicsr import build_model

print(os.path.abspath(os.curdir))
opt = yaml.safe_load(open("pretrained_models/RCAN/train_RCAN_x2.yml", 'r'))
opt['is_train'] = opt['dist'] = False
opt['path']['models'] = opt['path']['training_states'] = "pretrained_models/RCAN/"

for scale in [2,3,4]:
    pt_path = f"pretrained_models/RCAN/RCAN_BIX{scale}.pt"
    official = torch.load(pt_path)

    opt['scale'] = opt['network_g']['upscale'] = scale
    model = build_model(opt)
    
    for attr in ['weight', 'bias']:
        model.net_g.conv_first.__setattr__(attr, Parameter(official[f'head.0.{attr}']))
        model.net_g.conv_after_body.__setattr__(attr, Parameter(official[f'body.10.{attr}']))
        model.net_g.upsample[0].__setattr__(attr, Parameter(official[f'tail.0.0.{attr}']))
        if scale == 4:
            model.net_g.upsample[2].__setattr__(attr, Parameter(official[f'tail.0.2.{attr}']))
        model.net_g.conv_last.__setattr__(attr, Parameter(official[f'tail.1.{attr}']))
        for g in range(10):
            for b in range(20):
                model.net_g.body[g].residual_group[b].rcab[0].__setattr__(attr, 
                    Parameter(official[f'body.{g}.body.{b}.body.0.{attr}']))
                model.net_g.body[g].residual_group[b].rcab[2].__setattr__(attr, 
                    Parameter(official[f'body.{g}.body.{b}.body.2.{attr}']))
                model.net_g.body[g].residual_group[b].rcab[3].attention[1].__setattr__(attr, 
                    Parameter(official[f'body.{g}.body.{b}.body.3.conv_du.0.{attr}']))
                model.net_g.body[g].residual_group[b].rcab[3].attention[3].__setattr__(attr, 
                    Parameter(official[f'body.{g}.body.{b}.body.3.conv_du.2.{attr}']))
            model.net_g.body[g].conv.__setattr__(attr, 
                Parameter(official[f'body.{g}.body.20.{attr}']))
    
    model.save(1000, 9999)
    os.remove("pretrained_models/RCAN/9999.state")
    os.rename(src="pretrained_models/RCAN/net_g_9999.pth", 
              dst=f"pretrained_models/RCAN/RCAN_x{scale}c64b20g10.pth")
    print(f"Scale x{scale}: pretrained_models/RCAN/RCAN_x{scale}c64b20g10.pth")