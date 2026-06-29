# 课程学习训练

每个课程由一个 Python 文件定义，训练时用 `--course` 显式加载。课程文件主要覆盖初始化条件和少量环境参数；课程之间手动衔接，推荐从上一课程的 `best.pt` 继续，并使用 `--reset-progress` 重置 optimizer 和 global step。

## 课程顺序

1. `curricula/c01_three_ready.py`：3 艘拖轮已就位，1 艘短距离接入。
2. `curricula/c02_two_ready.py`：2 艘已就位，2 艘中等距离接入，仍使用 1.0 s hold，避免 ready 数量变化和 hold 时长同时加难。
3. `curricula/c03_one_ready.py`：1 艘已就位，3 艘从更宽分布接入，hold 提升到 1.5 s。
4. `curricula/c04_zero_ready.py`：0 艘已就位，覆盖 stern gate、side lane、outer slot 的近/中程初始化；rear lane 与 opposite stern 留给后续更难课程。
5. `curricula/strict_pos/c04_pos10m.py`：最终严格阶段，`pos_tol_m=10`，ready count 在 `0..4` 中随机采样。

## 推荐命令

```bash
python3 scripts/train.py \
  --course curricula/c01_three_ready.py \
  --run-name c01_three_ready

python3 scripts/train.py \
  --course curricula/c02_two_ready.py \
  --resume checkpoints/c01_three_ready/best.pt \
  --reset-progress \
  --critic-warmup-updates 5 \
  --run-name c02_two_ready

python3 scripts/train.py \
  --course curricula/c03_one_ready.py \
  --resume checkpoints/c02_two_ready/best.pt \
  --reset-progress \
  --critic-warmup-updates 5 \
  --run-name c03_one_ready

python3 scripts/train.py \
  --course curricula/c04_zero_ready.py \
  --resume checkpoints/c03_one_ready/best.pt \
  --reset-progress \
  --critic-warmup-updates 5 \
  --run-name c04_zero_ready
```

`--total-steps` 可以覆盖课程文件中的默认步数。其他 CLI 覆盖项在课程配置之后应用，例如 `--hold-time`、`--pos-tol`、`--heading-tol-deg` 和 `--speed-tol` 会覆盖课程文件里的对应环境参数。

## 严格到位阈值课程

当前 c04 的默认 `pos_tol_m=140` 只适合作为宽松引导指标。若目标是严格到位，需要继续收紧位置成功阈值。使用 `scripts/train_strict_pos_curriculum.py` 可以按位置阈值逐级训练，默认难度为：

```text
pos_tol: 140 -> 120 -> 100 -> 80 -> 60 -> 40 -> 20 -> 10 m
hold_time: 2 s
final target: success >= 90%, collision <= 8%
```

每一级也有独立课程文件，便于从 0 复现或手动单阶段训练：

```text
curricula/strict_pos/c04_pos140m.py
curricula/strict_pos/c04_pos120m.py
curricula/strict_pos/c04_pos100m.py
curricula/strict_pos/c04_pos80m.py
curricula/strict_pos/c04_pos60m.py
curricula/strict_pos/c04_pos40m.py
curricula/strict_pos/c04_pos20m.py
curricula/strict_pos/c04_pos10m.py
```

从已有 c04 权重继续训练严格课程：

```bash
PYTHONPATH=. python3 scripts/train_strict_pos_curriculum.py \
  --run-prefix strict_c04_pos10 \
  --initial-resume checkpoints/recover_c04_refine_lr2e5/best.pt \
  --stage-threshold 0.80 \
  --target-success 0.90 \
  --max-final-collision 0.08 \
  --success-bc-coef 0.10 \
  --eval-workers 1 \
  --torch-threads 1
```

脚本会先评估传入的 resume checkpoint；如果它已经满足当前阶段门槛，就跳过该阶段训练，避免把已有能力训练坏。阶段门槛同时检查 success 和 collision，避免把“能到位但高碰撞”的 checkpoint 传给下一阶段。最终通过条件只看 `pos_tol=10m, hold=2s` 的独立评估；中间宽松阈值通过不代表严格目标已经完成。最终 10 m 课程会把 `tug_init_mixed_ready_counts` 设为 `(0,1,2,3,4)`，用于覆盖任意 ready 数量。

严格位置课程文件本身已经写入 `reward_precision_*` 和 `reward_near_hold_*`，脚本默认使用课程文件中的奖励参数；只有显式传入同名 CLI 参数时才覆盖。`--success-bc-coef` 是训练侧设置：只对 rollout 中成功 episode 的采样动作增加一个小的 deterministic mean 行为克隆损失，用来减少“随机采样成功、确定性评估失败”的差距。
