# Design: 文档与代码全面对齐（清扫过时内容）

**Date:** 2026-08-06  
**Status:** Completed (historical); reward `rw_*` ablation docs removed in a later cleanup  
**Approach:** 手术式对齐（删死代码 + 文档按真相源修订）

## 1. Goal

使仓库文档与当前可运行主路径一致：删除课程学习、外部航路等已废弃能力的文档与代码残留；主文档默认值、模块列表、CLI、测试索引与代码真相源对齐。

## 2. Decisions (from brainstorming)

| 议题 | 选择 |
|------|------|
| 范围 | 全面审计：文档 + 与文档矛盾的遗留代码/测试 |
| 课程学习 | 全部删除（CLI、测试、脚本、文档叙述） |
| `route_planner` | 删除模块与相关测试；文档不再提路径规划 |
| `docs/superpowers/` | 只保留仍对应现有代码的设计/计划 |

## 3. Truth sources

文档内容必须以这些为准（冲突时改文档或删死代码，不以旧文档为准）：

| 主题 | 真相源 |
|------|--------|
| 任务阶段 / 成功判定 | `env/formation_env.py`、`config.EnvConfig`（`hold_time_s`、`track_horizon_s`、tol） |
| 观测维度 | `env/obs_spec.py` → `ObservationSpec` |
| 奖励 | `env/reward.py`、`EnvConfig` 奖励字段 |
| 奖励 preset | `config.REWARD_PRESETS`、`apply_reward_preset`、`--reward-preset` |
| Actor 架构 | `rl/actor.py`、`rl/temporal.py`、`--arch` |
| 训练 / TB | `scripts/train.py` |
| 初始化半径 | `env/init.py`、`tug_init_radius_m`、`--init-radius` |

## 4. Delete list

### 4.1 Documents

- `docs/curriculum_training.md`（工作区已删除，提交中保持删除）

### 4.2 Code / scripts / tests

| Path | Reason |
|------|--------|
| `env/route_planner.py` | 未接线；`FormationEnv` 不调用 |
| `tests/test_curricula.py` | 依赖已移除的 `curricula` 包 |
| `tests/test_reward_route_progress.py` | 依赖已移除的 route 奖励字段 |
| `tests/test_reward_precision.py` | `EnvConfig` 已无 `reward_precision_*` |
| `scripts/evaluate_ready_counts.py` | 已删除（保持） |
| `scripts/reproduce_c04_curriculum.py` | 已删除（保持） |
| `scripts/train_strict_pos_curriculum.py` | 已删除（保持） |

### 4.3 CLI / train wiring

从 `scripts/train.py` 移除：

- `curricula` 可选 import（`CourseSpec` / `apply_course` / `load_course`）
- `--course` 参数
- `course_spec` / `course_metadata` 加载与打印
- TensorBoard `writer.add_text("course", ...)`
- checkpoint payload 中的 `"course"` 字段写入

不保留任何 course 占位或「模块不可用」报错路径。

## 5. Document updates

### 5.1 Keep as living docs (edit in place)

| Doc | Changes |
|-----|---------|
| `README.md` | 索引与命令只反映现有脚本；无 curriculum / route |
| `docs/architecture.md` | 去掉 route「未接线遗留」、`--course` 说明、历史测试免责；模块/测试表对齐现状 |
| `docs/task_spec.md` | 保持 Approach→Capture→Track；去掉多余「旧默认 200 m」叙事若与主路径无关可压缩为一句远距复现说明 |
| `docs/observation_space.md` | 与 `ObservationSpec` 一致；可保留 schema v1→v2 迁移注记（仍描述现有 93 维布局）；删除任何航路/waypoint 观测描述（若有） |
| `docs/reward_function.md` | 与 `FormationRewardComputer` / `EnvConfig` 一致；无 precision/route 项 |
| `docs/arch_ablation.md` | 与 `--arch` / `build_actor` 一致 |
| `docs/tensorboard_metrics.md` | 与当前 `train.py` 写入的标量一致；去掉对 `course` 文本面板的现行描述（可一句说明旧 run 可能含历史标签） |

### 5.2 Superpowers archive policy

| Path | Action |
|------|--------|
| 奖励 `rw_*` preset 消融 design/plan | **已删除**（结构性重设计后清空） |
| 其它无对应代码的 superpowers 文档 | 删除 |

主文档索引（README）**不**强制链接 `docs/superpowers/`；架构消融协议见 `arch_ablation.md`；奖励 preset 映射见 `config.REWARD_PRESETS`。

## 6. Out of scope

- 不重写物理模型或奖励公式本身
- 不实现 GRU/LSTM（文档可继续标「接口预留」）
- 不删除 `simulator/`、操纵性测试、可视化脚本
- 不改 TensorBoard 历史 event 文件
- 不强制提交用户未要求的无关工作区改动（如 `outputs/`、`.playwright-cli/`）

## 7. Verification

1. 全库搜索无残留：`curricula`、`--course`、`RoutePlanner`、`route_planner`、`reward_precision`、`route_progress`（允许本 design/plan 历史叙述中出现「已删除」语境时尽量避免；代码与主 docs 必须为零）
2. 运行保留测试：  
   `pytest tests/test_actor_arch.py tests/test_attention.py tests/test_init_radius.py tests/test_observation_spec.py tests/test_reward_cpa.py tests/test_reward_presets.py tests/test_track_phase.py -q`  
   （`test_maneuvers.py` 可选、较慢）
3. `python scripts/train.py --help` 无 `--course`；有 `--arch`、`--reward-preset`、`--init-radius`
4. README 文档表中的链接文件均存在

## 8. Success criteria

- 新读者从 README → 主 docs 不会看到已移除的 curriculum / 外部航路能力
- 死测试与死模块不再导致 CI/本地 pytest 噪音
- 主文档默认值与 `config.py` / `obs_spec.py` 一致
