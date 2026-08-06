# GPU 并行环境仿真设计

日期：2026-08-07
状态：已批准，实现中
范围：旁路 Batched PyTorch `CudaVecEnv`，与 CPU `FormationEnv` 语义对齐；双后端长期共存。

## 1. 目标与决策

| 项 | 选择 |
|----|------|
| 目标 | 完整可训练 CudaVecEnv（v1） |
| 路线 | 旁路实现，不改写 OOP `FormationEnv` / Subproc |
| 共存 | `--env-backend {sync,subproc,cuda}` |
| 精度 | 对照：CPU + float64；训练：CUDA + float32 |
| 默认（扫参后） | `env_backend=cuda`、`num_envs=12288`、`minibatch_size=65536`、`rollout_steps=128`、`total_steps=50_000_000` |

明确不做（v1）：Isaac/Brax/Warp；删除 CPU 路径；改奖励/观测语义；以 AMP/`torch.compile` 为验收条件。

## 2. 架构

```
scripts/train.py → make_vec_env
  ├─ sync / subproc → FormationEnv (oracle)
  └─ cuda → CudaVecEnv → physics/batched + env/gpu
tests/parity 同时对照两侧
```

### 新增模块

- `physics/batched/`：`geometry.py`、`tugboat.py`、`ship.py`（真正 GPU/批张量步进）
- `env/gpu/`：`state.py`、`reward.py`、`terminate.py`、`observer.py`、`reset.py`、`vec_env.py`
- v1 实现策略：CUDA 上 **动力学 + reward/obs/终止** 均走 `env/gpu/batched_step.py` 设备批处理；float64+CPU 对照测试可回退 oracle 路径。
- `scripts/bench_env_throughput.py`：扫 `num_envs`
- `tests/parity/`：L0–L2 对照

## 3. VecEnv 契约（冻结）

与 `SyncVecEnv` 对齐：

- `reset() → obs (N, K, obs_dim) float32`
- `step(actions (N,K,4)) →` 9-tuple：
  `obs, rewards, dones, infos, ep_infos, terminated, truncated, terminal_obs_local, terminal_global`
- done 后返回的 `obs` 为 **reset 后**；`terminal_*` 为 **reset 前**（truncated GAE）
- `infos`: `list[dict]` 长度 N（含 `reward_components`、`success`/`capture`/`collision`/`terminated`/`truncated` 等）
- `get_global_state()` / 等价：`(N, global_state_dim)`，cuda 路径同设备

默认维度（`EnvConfig` 默认）：obs **93** / agent；global **82**（K=4）。

## 4. 阶段与终止语义（冻结）

与 `FormationEnv` 一致：

- Approach → Capture（全体 in-zone 满 `hold_time_s`）→ Track（再连续 in-zone 满 `track_horizon_s` → success）
- 碰撞：拖轮-大船 / 拖轮-拖轮 → terminated + culprit/bystander 终端罚
- timeout：`max_episode_steps` → truncated
- in-zone 计数：入区 +1 / 出区 −2（与 CPU 相同）

## 5. 状态 SoA

| 组 | 形状 | 内容 |
|----|------|------|
| 拖轮刚体 | `(N,K,3)` | eta, nu |
| 执行器 | `(N,K,4)` ×2 | cmd / actual |
| 缓存 | `(N,K,3)` | last_tau, last_nu_dot |
| 大船 | `(N,*)` | 位姿、速度、目标、resample 计时 |
| Episode | `(N,)` / `(N,K,*)` | step、phase、in_zone、prev_*、hist、capture/track |
| History | `(N,K,H,6)` / `(N,K,H,4)` | motion / action |
| 分配 | `(N,K)` | tug_to_slot |

## 6. Parity 容差档位

| 层级 | 内容 | 标准 |
|------|------|------|
| L0 | 单拖轮/大船固定动作序列 | float64 CPU：`rtol=1e-10`, `atol=1e-12`（可按实测微调并写死） |
| L1 | 同快照 reward/obs/phase/collision | 同上；离散事件完全一致 |
| L2 | N∈{1,4,8} 短轨迹含 reset | float64 对齐对齐；float32 CUDA：`rtol=1e-4` 且事件一致 |
| L3 | 训练冒烟 ≥1 update + 1 eval | 不崩溃；有限 loss/sps |

原则：先 float64 锁语义，再 CUDA float32。RNG 可注入或录制 CPU 序列喂 GPU。

## 7. num_envs 扫参

`scripts/bench_env_throughput.py`：

1. 固定 rollout 长度，扫 `num_envs` 直至 OOM 或 sps 平台
2. 记录 sps、显存；短训看 update 是否成新瓶颈
3. 推荐值写入文档；默认 backend 仍为 `subproc` 除非数据明确支持改默认

## 8. 验收

- L0–L2 绿；L3 冒烟绿
- CPU sync/subproc 仍可训练
- 相对 subproc 训练段 sps 明显提升（目标方向 ≥2×，不硬卡）
- 扫参给出推荐 `num_envs`

## 9. 实现后记（2026-08-07）

- Parity：`tests/parity/` L0–L2 通过（float64 CPU 可走 oracle）；`train.py --env-backend cuda` L3 冒烟通过。
- 实现：`env/gpu/batched_step.py` 设备端批处理 reward/终止/obs/global；CUDA 训练默认走快路径。float64+CPU 仍可对照 oracle。
- 吞吐（`bench_env_throughput.py`，rollout_steps=64）：
  - `cuda` 快路径：N=64/128/256 → sps≈7.5k / 14.9k / 27.2k
  - `subproc` 对照：N=64 → sps≈15.6k
  - N≥128 时 cuda 环境步进已追上/超过同机 subproc 量级；完整 MAPPO 还需结合 update/显存再扫。
- 扫参峰值已写入 `PPOConfig` / CLI 默认：`cuda` + `num_envs=12288` + `minibatch=65536`（细扫平台 ≈53–54k sps；显存 ≈8.5GB/24GB）。弱机器用 CLI 下调。
