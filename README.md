# 多智能体拖轮编队强化学习

项目说明见 [docs/README_RL.md](docs/README_RL.md)。

## 目录结构

```
config.py            全部超参数
physics/             拖轮/大船动力学模型
env/                 多智能体编队环境
rl/                  MAPPO 算法
utils/               通用工具
scripts/             入口脚本（train / visualize）
tests/               测试
docs/                设计文档
memory/              实验记录
```

## 快速开始

```bash
pip install torch numpy pygame tensorboard scipy

# 训练
python scripts/train.py

# 可视化
python scripts/visualize.py --ckpt checkpoints/<run_name>/best.pt

# 动力学操纵性试验
python tests/test_maneuvers.py
```
