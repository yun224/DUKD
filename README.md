# DUKD: Data Upcycling Knowledge Distillation for Image Super-Resolution

This repository is the official PyTorch implementation of [DUKD: Data Upcycling Knowledge Distillation for Image Super-Resolution](https://arxiv.org/abs/2309.14162).

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2309.14162)

---

![Framework of DUKD](DUKD-Framework.png)

> Knowledge distillation (KD) compresses deep neural networks by transferring task-related knowledge from cumbersome pre-trained teacher models to compact student models. However, current KD methods for super-resolution (SR) networks overlook the nature of SR task that the outputs of the teacher model are noisy approximations to the ground-truth distribution of high-quality images (GT), which shades the teacher model's knowledge to result in limited KD effects. To utilize the teacher model beyond the GT upper-bound, we present the Data Upcycling Knowledge Distillation (DUKD), to transfer the teacher model’s knowledge to the student model through the upcycled in-domain data derived from training data. Besides, we impose label consistency regularization to KD for SR by the paired invertible augmentations to improve the student model's performance and robustness. Comprehensive experiments demonstrate that the DUKD method significantly outperforms previous arts on several SR tasks.

---

## Environment

Install dependencies

```bash
# We use BasicSR for distillation.
pip install basicsr
pip install numpy 
pip install torch
```

## Data

We use DIV2K for training which can be downloaded from [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Four benchmark datasets are used as testing sets which can be downloaded from [https://cv.snu.ac.kr/research/EDSR/benchmark.tar](https://cv.snu.ac.kr/research/EDSR/benchmark.tar).

The default data storage format is
```
/cache/SR/
├── DIV2K
│   ├── DIV2K_train_HR
│   ├── DIV2K_train_LR_bicubic
│   │   ├── X2
│   │   │   ├── 0001x2.png
│   │   │   ├── 0002x2.png
│   │   │   ├── ...
│   │   │   └── 0800x2.png
│   │   ├── X3
│   │   └── X4
└── test
    ├── BSDS100
    │   ├── HR
    │   └── LR_bicubic
    │       ├── X2
    │       ├── X3
    │       └── X4
    ├── DIV2K_Val
    ├── Set14
    ├── Set5
    └── Urban100
```

## Train
```bash
python -c "import models; __import__('basicsr').train.train_pipeline('./')" -opt options/EDSR/dukd_edsr_x2c256b32_c64b32_22_zo_w51_n3.yml
# Before the formal training, you may run in the --debug mode to see if everything is OK. 
python -c "import models; __import__('basicsr').train.train_pipeline('./')" -opt options/EDSR/dukd_edsr_x2c256b32_c64b32_22_zo_w51_n3.yml --debug
```

More training configs can be found in `./options`. 


Teacher model **checkpoints** can be downloaded from 
- EDSR: [https://github.com/XPixelGroup/BasicSR/blob/master/docs/ModelZoo.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/ModelZoo.md).
    - EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth
    - EDSR_Lx3_f256b32_DIV2K_official-3660f70d.pth
    - EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth
- RCAN: [https://github.com/yulunzhang/RCAN](https://github.com/yulunzhang/RCAN)
    - RCAN_BIX2.pt
    - RCAN_BIX3.pt
    - RCAN_BIX4.pt
    - Checkpoint conversion: our code cannot directly use the downloaded RCAN checkpoints due to the inconsistency of network components' naming rule. A simple checkpoint conversion script is provided.
    ```bash
    python pretrained_models/RCAN/rcan_ckpt_conversion.py
    # Test the converted checkpoint
    python -c "__import__('basicsr').test.test_pipeline('./')" -opt options/test/test_RCAN_x4c64b6g10.yml

    > 2023-10-20 16:50:36,813 INFO: Validation Set5
    >      # psnr: 32.6388        Best: 32.6388 @ RCAN_x4c64b6g10 iter
    >      # ssim: 0.9002 Best: 0.9002 @ RCAN_x4c64b6g10 iter
    > 2023-10-20 16:50:41,530 INFO: Validation Set14
    >      # psnr: 28.8512        Best: 28.8512 @ RCAN_x4c64b6g10 iter
    >      # ssim: 0.7885 Best: 0.7885 @ RCAN_x4c64b6g10 iter
    > ...
    ```
- SwinIR: [https://github.com/JingyunLiang/SwinIR/releases](https://github.com/JingyunLiang/SwinIR/releases)
    - 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
    - 001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth
    - 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth


## Test
Student models are periodically evaluateded on four testing sets during training. To evaluate a model after training, prepare a testing config file and the testing command is like

```bash
python -c "__import__('basicsr').test.test_pipeline('./')" -opt options/test/test_RCAN_x4c64b6g10.yml
```
## Citations
```
@article{zhang2023data,
  title={Data Upcycling Knowledge Distillation for Image Super-Resolution},
  author={Zhang, Yun and Li, Wei and Li, Simiao and Chen, Hanting and Tu, Zhijun and Wang, Wenjia and Jing, Bingyi and Lin, Shaohui and Hu, Jie},
  journal={arXiv preprint arXiv:2309.14162},
  year={2023}
}
```
