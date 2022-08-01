# Human-in-the-loop Deep Reinforcement Learning (Hug-DRL)

This repo is the implementation of the paper "Human-in-the-Loop Deep Reinforcement Learning with Application to Autonomous Driving".

[**Human-in-the-Loop Deep Reinforcement Learning with Application to Autonomous Driving**](https://arxiv.org/abs/2104.07246) 
> Jingda Wu, Zhiyu Huang, Chao Huang, Zhongxu Hu, Peng Hang, Yang Xing, Chen Lv

## Getting started
1. Install the CARLA simulator (0.9.7), with referring to
https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation

2. Install the dependent package
```shell
pip install -r requirements.txt
```
3. Run the training procedure
```
python train_offline.py
```

## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{WU2022,
title = {Toward human-in-the-loop AI: Enhancing deep reinforcement learning via real-time human guidance for autonomous driving},
journal = {Engineering},
year = {2022},
issn = {2095-8099},
doi = {https://doi.org/10.1016/j.eng.2022.05.017},
author = {Jingda Wu and Zhiyu Huang and Zhongxu Hu and Chen Lv},
}
```

## License
This repo is released under GNU GPLv3 License.
