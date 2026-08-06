# GPU/CPU 环境采样一致性修复设计

日期：2026-08-07  
状态：已批准，待实施  
范围：`CudaVecEnv` 真实 batched 快路径与 CPU `FormationEnv` 的奖励、随机重采样、reset 和 rollout parity。

## 1. 目标

修复已经复现的两项语义差异：拖轮间 CPA 奖励方向错误，以及 CUDA 船舶重采样没有推进对应 CPU shadow environment 的 NumPy RNG。修复后，同一初始种子和动作序列应满足：

- 离散事件（capture、success、collision、terminated、truncated）完全一致；
- 奖励在 CUDA float32 合理数值容差内一致，不再出现 CPA 方向导致的 O(1) 偏差；
- 船舶重采样使用与 CPU 相同的逐环境随机序列；
- episode reset 后的 CPU 采样状态继续一致；
- 不改变现有奖励定义、观测 schema 或默认训练 backend。

## 2. 已确认根因

### 2.1 拖轮间 CPA 相对位置反号

batched reward 已将 `pair_dx/pair_dy` 构造为 `other - self`，但调用 `_cpa_risk` 时再次取负，而 `rvx/rvy` 仍为 `other_velocity - self_velocity`。位置和速度使用相反约定，使远离目标被当成接近目标。

修复只移除 CPA 调用处的位置负号；邻近距离惩罚、船舶 CPA 和 CPU oracle 不变。

### 2.2 CUDA RNG 与 shadow environment RNG 分叉

CPU fast path 从各 `FormationEnv.ship.rng` 取样；CUDA fast path 改用全局 `torch.rand`。因此：

1. 非退化船速范围下，第一次重采样后船舶轨迹立即分叉；
2. 即使默认船速固定为 1.0，CPU RNG 已消费随机数而 CUDA shadow RNG 未消费；
3. 下一次 episode reset 使用不同 RNG 状态，初始观测显著分叉。

## 3. 设计

### 3.1 CPA 修复

`FastBatchedStep.compute_rewards_batched` 向 `_cpa_risk` 传递同一约定的相对位置和相对速度：均为 `other - self`。增加 fast-path 奖励分量回归测试，直接比较 `p_tug_collision` 和最终 reward。

### 3.2 稀疏 CPU 预采样、GPU 执行

`CudaVecEnv` 在 CPU 和 CUDA fast path 都把 shadow environments 传给 `dynamics_step`。仅当某行的船舶重采样计时器到期时：

1. 按环境索引升序访问对应 `ship.rng`；
2. 按 CPU `LargeShipModel.step` 相同顺序抽取 `u_target`、下一重采样间隔；
3. 一次性转换并上传样本张量；
4. batched ship step 仅对到期行应用样本。

重采样每 20–40 秒才发生一次，允许此处发生一次设备同步和小批量 CPU→GPU 传输。正常环境 step 不增加 Python per-environment 循环。

船舶的 `time_to_resample` 独立使用 float64 张量，避免 float32 累计减法改变跨后端的触发步。船舶位置、速度和奖励仍使用训练 dtype（CUDA 默认 float32）。

### 3.3 TCPA 观测容差

不改变邻居 TCPA 的观测定义。该特征包含除以很小相对速度平方的运算，float32 状态误差会被明显放大。Parity 测试采用字段级口径：

- TCPA 以外的局部观测：`rtol=1e-4, atol=1e-4`；
- 邻居 TCPA：`rtol=1e-4, atol=5e-2`；
- global state：`rtol=1e-4, atol=1e-4`；
- 所有离散事件：精确相等。

这只调整验证口径，不修改训练输入。

## 4. 测试策略

遵循测试先行：

1. 新增 fast-path CPA 回归，修复前必须在 `p_tug_collision` 上失败；
2. 新增真实 CUDA 重采样/reset 回归，使用非退化船速范围和强制到期计时器，修复前必须在 reset 观测上失败；无 CUDA 时显式 skip；
3. 新增真实 fast-path 短 rollout，对 N=1/4/8 检查奖励、分字段观测、global state 和离散事件；
4. 复跑 `tests/parity/` 与完整测试集；
5. 运行小规模 throughput smoke，确认稀疏预采样没有改变正常热路径。

## 5. 非目标

- 不追求 CPU float64 与 CUDA float32 bitwise 相等；
- 不把所有随机初始化迁移到 GPU；
- 不修改奖励权重、CPA 定义或观测 schema；
- 不顺带重构 `FastBatchedStep`；
- 不改变默认 `subproc` backend。

## 6. 验收标准

- 新增回归测试在修复前失败、修复后通过；
- `tests/parity/` 全部通过，其中至少一条测试实际执行 CUDA fast path；
- CPA 回归中的 reward 差异降至 `1e-4` 量级以内；
- 强制重采样并 timeout 后，CPU/GPU reset 观测完全相同；
- N=1/4/8 rollout 的离散事件完全一致，连续量满足第 3.3 节容差；
- 完整测试集无新增失败。
