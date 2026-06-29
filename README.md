# 多智能体拖轮编队强化学习

项目说明见 [docs/README_RL.md](docs/README_RL.md)。

## 训练的核心目标
无论初始化是几艘船到位，拖轮智能体可以全部在pos_tol_m = 10 m的时候稳定伴航大船

## 目录结构

```
config.py            全部超参数
physics/             拖轮/大船动力学模型
env/                 多智能体编队环境
rl/                  MAPPO 算法
utils/               通用工具
simulator/           罗技 G29 方向盘手动驾驶仿真
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

# 从无权重复现到 10 m mixed-ready 最终课程
RUN=final10_$(date +%Y%m%d_%H%M%S)
python3 scripts/reproduce_c04_curriculum.py \
  --run-prefix "$RUN" \
  --stage-attempts 2 \
  --target-success 0.80 \
  --max-final-collision 0.08 \
  --eval-workers 1 \
  --torch-threads 1

python3 scripts/train_strict_pos_curriculum.py \
  --run-prefix "${RUN}_strict" \
  --initial-resume "checkpoints/${RUN}_c04_zero_ready_refine/best.pt" \
  --no-intermediate-gates \
  --stage-threshold 0.80 \
  --target-success 0.90 \
  --max-final-collision 0.08 \
  --success-bc-coef 0.10 \
  --eval-workers 1 \
  --torch-threads 1

# 查看训练曲线（默认日志目录 runs/，可用 --run-name 指定 run 名）
tensorboard --logdir runs

# 可视化
python scripts/visualize.py --ckpt checkpoints/<run_name>/best.pt

# 动力学操纵性试验
python tests/test_maneuvers.py

# 手动驾驶仿真（罗技 G29 方向盘）
# 方向盘转角同时控制左右两舵方位角；油门踏板控左桨、刹车踏板控右桨转速；+/- 调仿真倍速
# 默认有一艘匀速直线大船，HUD 显示拖轮相对大船的纵/横向偏移与速度差，便于测试伴航控制
python -m simulator
python -m simulator --speed 4 --mpp 0.4          # 4 倍速、放大视角
python -m simulator --ship-speed 2.0             # 大船匀速 2 m/s
python -m simulator --no-ship                     # 不显示大船
python -m simulator --no-wheel                    # 无方向盘时用键盘控制
python -m simulator --throttle-axis 1 --brake-axis 2  # 覆盖踏板轴（按 D 看轴调试面板）
```

方向盘换挡拨片：按住左拨片让左桨反转、右拨片让右桨反转（增强原地回转/横移灵活性）；HUD 会显示 `[反转]`，按钮索引可在 `D` 调试面板查看或用 `--paddle-left-button/--paddle-right-button` 覆盖。

仿真内快捷键：`Space` 暂停 / `R` 重置 / `+`-`-` 调倍速 / `D` 切换轴/按钮调试面板 / `C` 校准踏板（踩满两踏板后再按 `C`）/ `Esc` 退出。键盘模式：方向键转向，`Q`/`Z` 调左桨，`E`/`C` 调右桨，按住左/右 `Shift` 反转左/右桨。
