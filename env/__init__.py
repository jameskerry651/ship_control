"""多智能体编队环境。"""

from env.formation_env import ACTION_DIM, FormationEnv
from env.obs_spec import ObservationSpec
from env.reward import FormationRewardComputer

__all__ = ["ACTION_DIM", "FormationEnv", "FormationRewardComputer", "ObservationSpec"]
