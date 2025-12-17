# TAROT
## Overview
This repository contains the official PyTorch implementation of **TAROT**, the algorithm proposed in the paper:
<br>
["TAROT: Towards Essentially Domain-Invariant Robustness
with Theoretical Justification"](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_TAROT_Towards_Essentially_Domain-Invariant_Robustness_with_Theoretical_Justification_CVPR_2025_paper.pdf)
<br>
by *Dongyoon Yang, Jihu Lee, and Yongdai Kim.*

The paper was published in CVPR 2025. 


## Robustly Pretrained Model
You can get robustly pretrained models at https://huggingface.co/madrylab/robust-imagenet-models.

## Run Example
* python3 tarot.py data/officehome -d OfficeHome -s Ar -t Cl -a resnet50 --eps 0.06274509 --step_size 0.01568 --pre_eps 1.0 --trade-off 0.5 --log logs/tarot/officehome_Ar2Cl_pre-eps_1.0_eps_16_trade-off_0.5 --gpu 0;
* python3 tarot.py data/officehome -d OfficeHome -s Ar -t Cl -a resnet50 --eps 0.06274509 --step_size 0.01568 --pre_eps 1.0 --trade-off 0.5 --log logs/tarot/officehome_Ar2Cl_pre-eps_1.0_eps_16_trade-off_0.5 --gpu 0 --phase test;
