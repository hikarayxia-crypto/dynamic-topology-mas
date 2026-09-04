"""共享图策略的推理智能体封装。"""

from __future__ import annotations

from typing import Hashable

import torch

from algorithms.shared_actor_critic import SharedGraphActorCritic
from core.action import Action
from core.agent import AgentAttributes, AgentState, BaseAgent
from core.observation import Observation


class SharedPolicyAgent(BaseAgent):
    """复用同一 Actor 参数进行分散执行的智能体。

    训练器会联合采样全部智能体以使用集中式 Critic；部署或验证时，每个实例只
    把自己的局部图观测交给共享 Actor，符合 CTDE 的分散执行要求。
    """

    def __init__(
        self,
        agent_id: Hashable,
        policy: SharedGraphActorCritic,
        attributes: AgentAttributes | None = None,
        state: AgentState | None = None,
        *,
        deterministic: bool = True,
    ) -> None:
        super().__init__(agent_id, attributes, state)
        self.policy = policy
        self.deterministic = deterministic
        self._observation: Observation | None = None

    def perceive(self, observation: Observation) -> Observation:
        """保存当前局部图观测。"""

        self._observation = observation
        return observation

    def decide(self, dt: float) -> Action:
        """用共享 Actor 产生二维连续动作。"""

        if self._observation is None or not self.state.active:
            timestamp = self.environment.current_time if self.environment else 0.0
            return Action.noop(self.id, timestamp=timestamp)
        with torch.no_grad():
            actions, _, _ = self.policy.act(
                [self._observation], deterministic=self.deterministic
            )
        return Action.continuous(
            self.id,
            actions[0].cpu().numpy(),
            timestamp=self._observation.timestamp,
            metadata={"policy": "shared_graph_actor"},
        )
