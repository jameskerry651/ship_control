"""C4 消融 #1：仅启用 P1（终端碰撞惩罚按责分配）。

基线 = 原始奖励（全员共担 −80、无团队项/势函数/streak、w_collision=5、无 cap），
本课程只把 bystander 惩罚降到 15，单独验证 P1 对 critic EV 与碰撞率的修复。
"""

from curricula.presets.ablations import ABLATION_COURSES

COURSE = ABLATION_COURSES["c04_abl_p1"]
