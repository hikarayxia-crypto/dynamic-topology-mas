"""不依赖训练的连续二维协同搜索规则基线。"""

from __future__ import annotations

import math
from typing import Hashable

import numpy as np

from core.action import Action
from core.agent import AgentAttributes, AgentState, BaseAgent
from core.observation import Observation


class RuleBasedSearchAgent(BaseAgent):
    """分带往返扫描并带邻居避碰的规则智能体。

    每个在线智能体根据当前索引领取一条水平搜索带，在左右边界间往返；每完成
    一次横向扫描就把搜索带向上移动一定距离。邻居过近时叠加排斥向量。该方法
    不读取未知目标坐标，可作为后续 GNN/MAPPO 的可解释非学习基线。
    """

    def __init__(
        self,
        agent_id: Hashable,
        attributes: AgentAttributes | None = None,
        state: AgentState | None = None,
        *,
        separation_distance: float = 1.5,
        separation_strength: float = 1.2,
    ) -> None:
        super().__init__(agent_id, attributes, state)
        if not math.isfinite(separation_distance) or separation_distance <= 0:
            raise ValueError("separation_distance 必须是正有限数值")
        if not math.isfinite(separation_strength) or separation_strength < 0:
            raise ValueError("separation_strength 必须是非负有限数值")
        self.separation_distance = float(separation_distance)
        self.separation_strength = float(separation_strength)
        self._observation: Observation | None = None
        self._horizontal_direction = 1.0
        self._completed_passes = 0

    def perceive(self, observation: Observation) -> Observation:
        """保存结构化观测，决策阶段仅使用可观测信息。"""

        self._observation = observation
        return observation

    def decide(self, dt: float) -> Action:
        """生成归一化二维速度指令。"""

        if self._observation is None or not self.state.active:
            return Action.noop(self.id, timestamp=self._timestamp())
        if self._observation.metadata.get("all_targets_discovered", False):
            return Action.noop(self.id, timestamp=self._observation.timestamp)

        command = self._sweep_command(self._observation, dt)
        return Action.continuous(
            self.id,
            command,
            timestamp=self._observation.timestamp,
            metadata={"policy": "rule_based_sweep"},
        )

    def _sweep_command(
        self,
        observation: Observation,
        dt: float,
        *,
        target_y: float | None = None,
    ) -> np.ndarray:
        """计算搜索带扫描与邻居排斥后的单位速度指令。

        参数为当前观测、时间步长和可选纵向搜索带中心；省略 ``target_y`` 时严格
        复用规则策略原有的分带公式。返回始终为有限的二维向量且范数不超过一，
        使补位策略能够只替换纵向带而不改变横向折返和避碰规则。
        """

        metadata = observation.metadata
        width, height = metadata["world_size"]
        position = np.asarray(metadata["position"], dtype=np.float64)
        agent_index = int(metadata["agent_index"])
        agent_count = max(1, int(metadata["agent_count"]))
        sensor_range = float(metadata["sensor_range"])
        boundary_tolerance = max(0.2, self.attributes.max_speed * dt)

        target_x = width if self._horizontal_direction > 0 else 0.0
        if abs(position[0] - target_x) <= boundary_tolerance:
            self._horizontal_direction *= -1.0
            self._completed_passes += 1
            target_x = width if self._horizontal_direction > 0 else 0.0

        # 初始分带减少重复覆盖；每次折返上移一个感知直径，逐步覆盖完整区域。
        base_lane = (agent_index + 0.5) * height / agent_count
        sweep_y = (base_lane + self._completed_passes * 2.0 * sensor_range) % height
        desired_y = sweep_y if target_y is None else float(target_y)
        desired = np.asarray([target_x - position[0], desired_y - position[1]])

        repulsion = np.zeros(2, dtype=np.float64)
        for neighbor in observation.neighbors:
            relative = np.asarray(
                neighbor.metadata.get("relative_position", [0.0, 0.0]),
                dtype=np.float64,
            )
            distance = float(neighbor.metadata.get("distance", np.linalg.norm(relative)))
            if 1e-9 < distance < self.separation_distance:
                closeness = 1.0 - distance / self.separation_distance
                repulsion -= (
                    relative / distance * closeness * self.separation_strength
                )
        desired += repulsion

        # 外部观测元数据可能含异常数值；控制输出仍须满足环境的有限动作契约。
        desired = np.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(desired))
        return desired / norm if norm > 1e-12 else np.zeros(2, dtype=np.float64)

    def _timestamp(self) -> float:
        return self.environment.current_time if self.environment is not None else 0.0
