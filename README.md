# Human-in-the-loop Deep Reinforcement Learning (Hug-DRL)

This is a code repository for the paper "Human-in-the-Loop Deep Reinforcement Learning with Application to Autonomous Driving"
#
[**Human-in-the-Loop Deep Reinforcement Learning with Application to Autonomous Driving**](https://arxiv.org/abs/2104.07246) 
> Jingda Wu, Zhiyu Huang, Chao Huang, Zhongxu Hu, Peng Hang, Yang Xing, Chen Lv

## Getting start
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
@misc{2104.07246,
Author = {Jingda Wu and Zhiyu Huang and Chao Huang and Zhongxu Hu and Peng Hang and Yang Xing and Chen Lv},
Title = {Human-in-the-Loop Deep Reinforcement Learning with Application to Autonomous Driving},
Year = {2021},
Eprint = {arXiv:2104.07246},
}
```

## License
This repo is released under MIT License.
