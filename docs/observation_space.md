# 拖轮编队控制观测状态空间

> 实现位置：`env/formation_env.py` 的 `_build_obs()` 与 `get_global_state()`  
> Actor 网络：`rl/actor.py` 的 `MAPPOActor`  
> 当前默认配置：`obs_history_k = 3`，`obs_ship_preview_times_s = (5.0, 10.0, 15.0)`

## 1. 坐标与符号约定

世界坐标系使用项目统一约定：`x` 指北，`y` 指东。船体系或拖轮自身坐标系的 `x` 轴指船首方向，`y` 轴指右舷方向。

世界系相对向量转到某个朝向为 `psi` 的本地坐标系时，使用：

$$
\begin{aligned}
x_{\mathrm{local}} &= \cos\psi \cdot \Delta x_{\mathrm{world}}
                  + \sin\psi \cdot \Delta y_{\mathrm{world}}, \\
y_{\mathrm{local}} &= -\sin\psi \cdot \Delta x_{\mathrm{world}}
                   + \cos\psi \cdot \Delta y_{\mathrm{world}}.
\end{aligned}
$$

拖轮 `i` 的状态记为：

$$
\eta_i = (x_i, y_i, \psi_i), \qquad
\nu_i = (u_i, v_i, r_i).
$$

其中 `psi` 是航向角，`u, v` 是拖轮自身体系下的纵向/横向速度，`r` 是偏航角速度。

## 2. Actor 局部观测，93 维

每个拖轮的 actor 输入是一个 93 维向量。环境仍返回扁平向量，actor 内部再切分为自身主观测和邻居 attention 输入。

| 索引范围 | 维数 | 名称 |
|---:|---:|---|
| 0-23 | 24 | 自身运动历史 |
| 24-39 | 16 | 执行机构命令历史 |
| 40-44 | 5 | 大船相对状态 |
| 45-50 | 6 | 大船轨迹前瞻 |
| 51-55 | 5 | 本 agent 目标 slot |
| 56-59 | 4 | 当前 route 目标与进度 |
| 60-62 | 3 | 最近船体边界向量与距离 |
| 63-92 | 30 | 邻居 attention 输入（碰撞风险特征） |

Actor 内部切片为：

$$
o_i^{\mathrm{own}} = o_i[0:63],
\qquad
o_i^{\mathrm{nbr}} = \mathrm{reshape}(o_i[63:93], 3, 10).
$$

## 3. 自身运动历史，24 维

历史长度为当前帧加过去 3 帧，共 4 帧。顺序是从新到旧：

$$
t,\; t-1,\; t-2,\; t-3.
$$

每一帧包含 6 个量。该块不再使用世界系绝对航向 `sin(psi), cos(psi)`，而是显式给出拖轮偏航角速度 `r`，减少旋转不变任务中的无关输入：

$$
h_i^\tau =
\left[
\frac{u_i^\tau}{5},
\frac{v_i^\tau}{5},
\frac{r_i^\tau}{0.5},
\frac{\Delta u_i^\tau}{5},
\frac{\Delta v_i^\tau}{5},
\frac{\Delta r_i^\tau}{0.5}
\right].
$$

差分项定义为：

$$
\Delta u_i^\tau = u_i^\tau - u_i^{\tau-1},
\qquad
\Delta v_i^\tau = v_i^\tau - v_i^{\tau-1},
\qquad
\Delta r_i^\tau = r_i^\tau - r_i^{\tau-1}.
$$

因此自身运动历史块为：

$$
o_{i,0:24}
=
\left[
h_i^t,\;
h_i^{t-1},\;
h_i^{t-2},\;
h_i^{t-3}
\right].
$$

reset 时，4 帧全部用初始状态填充，所有差分项为 0。

## 4. 执行机构命令历史，16 维

动作是 4 维归一化推进器命令：

$$
a_i^\tau =
\left[
n_{L,i}^\tau,\;
n_{R,i}^\tau,\;
\delta_{L,i}^\tau,\;
\delta_{R,i}^\tau
\right],
\qquad
a_i^\tau \in [-1, 1]^4.
$$

动作历史同样按从新到旧排列：

$$
o_{i,24:40}
=
\left[
a_i^t,\;
a_i^{t-1},\;
a_i^{t-2},\;
a_i^{t-3}
\right].
$$

其中 `n_L, n_R` 是左右推进器转速命令，`delta_L, delta_R` 是左右推进器方位角命令。reset 时，历史全部用初始动作填充。

## 5. 大船相对状态，5 维

大船中心相对拖轮 `i` 的世界系位移为：

$$
\Delta p_{s,i}
=
\begin{bmatrix}
x_s - x_i \\
y_s - y_i
\end{bmatrix}.
$$

将其投影到拖轮自身坐标系，得到：

$$
\begin{bmatrix}
\Delta x_{s,i}^{\mathrm{ego}} \\
\Delta y_{s,i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)\Delta p_{s,i}.
$$

大船相对状态块为：

$$
o_{i,40:45}
=
\left[
\frac{\Delta x_{s,i}^{\mathrm{ego}}}{100},
\frac{\Delta y_{s,i}^{\mathrm{ego}}}{100},
\frac{u_s}{3},
\sin(\psi_s-\psi_i),
\cos(\psi_s-\psi_i)
\right].
$$

这里 `u_s` 是大船自身船体系纵向速度。当前大船模型不使用横向速度，`v_s` 不再进入 actor 观测。

## 6. 大船轨迹前瞻，6 维

默认前瞻时间为：

$$
\tau_1 = 5\,\mathrm{s}, \qquad
\tau_2 = 10\,\mathrm{s}, \qquad
\tau_3 = 15\,\mathrm{s}.
$$

对每个前瞻时间 `tau`，环境使用大船当前速度估计未来大船中心位置。如果大船偏航角速度近似为 0：

$$
\begin{aligned}
x_s(\tau) &= x_s + v_{x,s}^{\mathrm{world}}\tau, \\
y_s(\tau) &= y_s + v_{y,s}^{\mathrm{world}}\tau.
\end{aligned}
$$

其中：

$$
\begin{aligned}
v_{x,s}^{\mathrm{world}} &= \cos\psi_s \cdot u_s - \sin\psi_s \cdot v_s, \\
v_{y,s}^{\mathrm{world}} &= \sin\psi_s \cdot u_s + \cos\psi_s \cdot v_s.
\end{aligned}
$$

当大船偏航角速度不为 0 时，环境按恒定船体系速度和恒定偏航角速度近似积分：

$$
\begin{aligned}
\Delta x_s^{\mathrm{body}}(\tau)
&=
\frac{
u_s\sin(r_s\tau)
+ v_s\left(\cos(r_s\tau)-1\right)
}{r_s},
\\
\Delta y_s^{\mathrm{body}}(\tau)
&=
\frac{
u_s\left(1-\cos(r_s\tau)\right)
+ v_s\sin(r_s\tau)
}{r_s}.
\end{aligned}
$$

该前瞻点再转到世界系，并相对当前拖轮投影到拖轮自身坐标系：

$$
\begin{bmatrix}
\Delta x_{s,i}^{\mathrm{ego}}(\tau) \\
\Delta y_{s,i}^{\mathrm{ego}}(\tau)
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
x_s(\tau) - x_i \\
y_s(\tau) - y_i
\end{bmatrix}.
$$

最终前瞻块为：

$$
o_{i,45:51}
=
\left[
\frac{\Delta x_{s,i}^{\mathrm{ego}}(\tau_1)}{100},
\frac{\Delta y_{s,i}^{\mathrm{ego}}(\tau_1)}{100},
\frac{\Delta x_{s,i}^{\mathrm{ego}}(\tau_2)}{100},
\frac{\Delta y_{s,i}^{\mathrm{ego}}(\tau_2)}{100},
\frac{\Delta x_{s,i}^{\mathrm{ego}}(\tau_3)}{100},
\frac{\Delta y_{s,i}^{\mathrm{ego}}(\tau_3)}{100}
\right].
$$

## 7. 本 agent 目标 slot，5 维

每个拖轮 `i` 被分配到一个固定目标 slot。目标 slot 的世界系位置与期望航向记为：

$$
s_i = (x_{\mathrm{slot},i}, y_{\mathrm{slot},i}, \psi_{\mathrm{slot},i}).
$$

slot 相对拖轮的位移投影到拖轮自身坐标系：

$$
\begin{bmatrix}
\Delta x_{\mathrm{slot},i}^{\mathrm{ego}} \\
\Delta y_{\mathrm{slot},i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
x_{\mathrm{slot},i} - x_i \\
y_{\mathrm{slot},i} - y_i
\end{bmatrix}.
$$

目标 slot 块为：

$$
o_{i,51:56}
=
\left[
\frac{\Delta x_{\mathrm{slot},i}^{\mathrm{ego}}}{100},
\frac{\Delta y_{\mathrm{slot},i}^{\mathrm{ego}}}{100},
\min\!\left(\frac{d_{\mathrm{slot},i}}{100}, 10\right),
\sin(\psi_{\mathrm{slot},i}-\psi_i),
\cos(\psi_{\mathrm{slot},i}-\psi_i)
\right].
$$

这 5 维是共享 actor 区分不同 agent 目标的关键输入。

## 8. 当前 route 目标与进度，4 维

环境为每艘拖轮维护一条从初始位置到目标 slot 的 route。actor 观测当前 route stage 对应的 waypoint，而不只看最终 slot。

当前 route 目标点相对拖轮的位移投影到拖轮自身坐标系：

$$
\begin{bmatrix}
\Delta x_{\mathrm{route},i}^{\mathrm{ego}} \\
\Delta y_{\mathrm{route},i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
x_{\mathrm{route},i} - x_i \\
y_{\mathrm{route},i} - y_i
\end{bmatrix}.
$$

route 块为：

$$
o_{i,56:60}
=
\left[
\frac{\Delta x_{\mathrm{route},i}^{\mathrm{ego}}}{100},
\frac{\Delta y_{\mathrm{route},i}^{\mathrm{ego}}}{100},
\frac{\mathrm{stage}_i}{N_{\mathrm{route},i}-1},
\frac{D_{\mathrm{remaining},i}}{500}
\right].
$$

其中 stage 归一化项会裁剪到 `[0, 1]`。

## 9. 最近船体边界向量与距离，3 维

为了让 actor 直接感知船体安全边界，环境计算拖轮到大船矩形碰撞 hull 的最近点。最近点先在大船船体系中计算，再转回世界系并投影到拖轮自身坐标系：

$$
\begin{bmatrix}
\Delta x_{\mathrm{hull},i}^{\mathrm{ego}} \\
\Delta y_{\mathrm{hull},i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
x_{\mathrm{hull},i}^{*} - x_i \\
y_{\mathrm{hull},i}^{*} - y_i
\end{bmatrix}.
$$

hull 块为：

$$
o_{i,60:63}
=
\left[
\frac{\Delta x_{\mathrm{hull},i}^{\mathrm{ego}}}{50},
\frac{\Delta y_{\mathrm{hull},i}^{\mathrm{ego}}}{50},
\frac{d_{\mathrm{hull},i}}{50}
\right].
$$

其中 `d_hull` 与碰撞检测使用的 hull 距离定义一致。

## 10. 邻居 attention 输入，30 维

当前默认 4 条拖轮。对拖轮 `i`，其邻居是其余 3 条拖轮，按 tug index 固定顺序排列。每个邻居从纯几何状态升级为碰撞风险特征，使本船特征（63 维）与邻居特征（3×10=30 维）维度更对等。

邻居 `j` 相对拖轮 `i` 的位置投影到拖轮 `i` 自身坐标系：

$$
\begin{bmatrix}
\Delta x_{j,i}^{\mathrm{ego}} \\
\Delta y_{j,i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
x_j - x_i \\
y_j - y_i
\end{bmatrix},
\qquad
d_{j,i} = \sqrt{(\Delta x_{j,i}^{\mathrm{ego}})^2 + (\Delta y_{j,i}^{\mathrm{ego}})^2}.
$$

方位角（邻居相对本船船首的夹角）：

$$
\theta_{j,i} = \operatorname{atan2}\!\left(\Delta y_{j,i}^{\mathrm{ego}}, \Delta x_{j,i}^{\mathrm{ego}}\right).
$$

相对速度同样投影到拖轮 `i` 自身坐标系（先把各船自身体系速度转到世界系，再做差并投影）：

$$
\begin{bmatrix}
\Delta u_{j,i}^{\mathrm{ego}} \\
\Delta v_{j,i}^{\mathrm{ego}}
\end{bmatrix}
=
R(-\psi_i)
\begin{bmatrix}
v_{x,j}^{\mathrm{world}} - v_{x,i}^{\mathrm{world}} \\
v_{y,j}^{\mathrm{world}} - v_{y,i}^{\mathrm{world}}
\end{bmatrix}.
$$

距离变化率（range rate，沿视线方向的相对速度投影，负值表示接近）：

$$
\dot d_{j,i}
=
\frac{
\Delta x_{j,i}^{\mathrm{ego}}\,\Delta u_{j,i}^{\mathrm{ego}}
+ \Delta y_{j,i}^{\mathrm{ego}}\,\Delta v_{j,i}^{\mathrm{ego}}
}{d_{j,i}}.
$$

最近会遇时刻 TCPA 与最近会遇距离 DCPA（基于匀速外推，过去的会遇裁剪为 0）：

$$
t^{\mathrm{CPA}}_{j,i}
=
\max\!\left(
-\frac{
\Delta x_{j,i}^{\mathrm{ego}}\,\Delta u_{j,i}^{\mathrm{ego}}
+ \Delta y_{j,i}^{\mathrm{ego}}\,\Delta v_{j,i}^{\mathrm{ego}}
}{
(\Delta u_{j,i}^{\mathrm{ego}})^2 + (\Delta v_{j,i}^{\mathrm{ego}})^2
},\;
0
\right),
$$

$$
d^{\mathrm{CPA}}_{j,i}
=
\left\|
\begin{bmatrix}
\Delta x_{j,i}^{\mathrm{ego}} + \Delta u_{j,i}^{\mathrm{ego}}\, t^{\mathrm{CPA}}_{j,i} \\
\Delta y_{j,i}^{\mathrm{ego}} + \Delta v_{j,i}^{\mathrm{ego}}\, t^{\mathrm{CPA}}_{j,i}
\end{bmatrix}
\right\|.
$$

当相对速度近似为 0（两船相对静止）时退化处理：$t^{\mathrm{CPA}}_{j,i}=60\,\mathrm{s}$，$d^{\mathrm{CPA}}_{j,i}=d_{j,i}$，$\dot d_{j,i}=0$。

每个邻居的 10 维风险特征为：

$$
b_{j|i}
=
\left[
\frac{\Delta x_{j,i}^{\mathrm{ego}}}{100},
\frac{\Delta y_{j,i}^{\mathrm{ego}}}{100},
\min\!\left(\frac{d_{j,i}}{100}, 10\right),
\sin\theta_{j,i},
\cos\theta_{j,i},
\frac{\Delta u_{j,i}^{\mathrm{ego}}}{5},
\frac{\Delta v_{j,i}^{\mathrm{ego}}}{5},
\frac{\dot d_{j,i}}{5},
\min\!\left(\frac{t^{\mathrm{CPA}}_{j,i}}{60}, 10\right),
\min\!\left(\frac{d^{\mathrm{CPA}}_{j,i}}{100}, 10\right)
\right].
$$

3 个邻居拼接为：

$$
o_{i,63:93}
=
\left[
b_{j_1|i},\;
b_{j_2|i},\;
b_{j_3|i}
\right].
$$

## 11. Actor attention 网络如何使用观测

Actor 将 93 维输入切成自身主观测和邻居观测：

$$
o_i^{\mathrm{own}} \in \mathbb{R}^{63},
\qquad
o_i^{\mathrm{nbr}} \in \mathbb{R}^{3\times 10}.
$$

自身编码器：

$$
e_i^{\mathrm{own}}
=
f_{\mathrm{own}}(o_i^{\mathrm{own}})
\in \mathbb{R}^{64}.
$$

邻居共享编码器：

$$
e_{j|i}^{\mathrm{nbr}}
=
f_{\mathrm{nbr}}(b_{j|i})
\in \mathbb{R}^{64}.
$$

Attention 使用自身特征生成 Query，邻居特征生成 Key 和 Value：

$$
q_i = W_Q e_i^{\mathrm{own}},
\qquad
k_{j|i} = W_K e_{j|i}^{\mathrm{nbr}},
\qquad
v_{j|i} = W_V e_{j|i}^{\mathrm{nbr}}.
$$

邻居权重为：

$$
\alpha_{j|i}
=
\frac{
\exp\left(q_i^\top k_{j|i}/\sqrt{64}\right)
}{
\sum_{\ell \neq i}
\exp\left(q_i^\top k_{\ell|i}/\sqrt{64}\right)
}.
$$

聚合后的威胁特征为：

$$
c_i
=
W_O
\left(
\sum_{j\neq i}\alpha_{j|i}v_{j|i}
\right)
\in \mathbb{R}^{64}.
$$

策略头输入为自身特征和威胁特征的拼接：

$$
z_i =
\left[
e_i^{\mathrm{own}},\;
c_i
\right]
\in \mathbb{R}^{128}.
$$

最终输出 4 维动作均值，动作分布仍是 tanh-squashed diagonal Gaussian。

## 12. Centralized Critic 全局状态，90 维

MAPPO critic 不使用 actor 的 93 维局部观测，而是使用 `get_global_state()` 构造的 canonical global state。默认 4 条拖轮时维度为：

$$
2 + 4\times 19 + 4\times 3 = 90.
$$

### 12.1 大船段，2 维

$$
s_{\mathrm{ship}}
=
\left[
\frac{u_s}{5},
\frac{\dot u_s}{0.2}
\right].
$$

当前大船模型直行且不使用横向速度、偏航角速度；船长和船宽默认不随机化，因此这些常量不再进入 critic。

### 12.2 每条拖轮段，19 维

对每条拖轮，critic 使用大船船体系下的位置、速度、相对朝向、执行器、动作、route 进度和船体距离。固定 `tug_to_slot = arange(n_tugs)` 下的 slot one-hot 是常量，已从 critic 状态中移除。

| 相对索引 | 维数 | 内容 |
|---:|---:|---|
| 0-1 | 2 | 拖轮在大船船体系下的位置，除以 100 |
| 2-3 | 2 | 拖轮世界速度投影到大船船体系，除以 5 |
| 4-5 | 2 | 拖轮航向相对大船航向的正弦和余弦 |
| 6 | 1 | 拖轮偏航角速度，除以 0.5 |
| 7-10 | 4 | 执行器实际值，已归一化 |
| 11-14 | 4 | 上一步动作 |
| 15 | 1 | route stage 进度 |
| 16 | 1 | route remaining，除以 500 |
| 17 | 1 | in-zone hold 进度 |
| 18 | 1 | 到大船船体的最近距离，除以 50 |

### 12.3 每条拖轮加速度 tail，3 维

每条拖轮在全局状态末尾还有 3 维拖轮自身加速度 tail：

$$
s_{i}^{\mathrm{acc}}
=
\left[
\frac{a_{x,i}^{\mathrm{ship}}}{1.0},
\frac{a_{y,i}^{\mathrm{ship}}}{1.0},
\frac{\dot r_i}{0.1}
\right].
$$

其中拖轮线加速度先从拖轮自身体系转到世界系，再投影到大船船体系。大船纵向加速度已放入大船段一次，不再在每条拖轮 tail 中重复。

## 13. 数值裁剪

Actor 观测和 critic 全局状态在返回前都会做统一裁剪：

$$
x \leftarrow \mathrm{clip}(x, -10, 10).
$$

这一步用于处理极端状态下的数值兜底，避免 NaN、Inf 或过大输入传入网络。
