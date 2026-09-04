"""多智能体环境使用的统一动作结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


class ActionType(str, Enum):
    """动作类型，兼容连续控制、离散决策和保持不动。"""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    NOOP = "noop"


@dataclass(frozen=True)
class Action:
    """单个智能体在一个时间步内提交的动作。

    参数:
        agent_id: 动作所属智能体。
        action_type: 连续、离散或空操作。
        value: 连续动作向量、离散动作编号或 ``None``。
        timestamp: 动作生成时的仿真时间。
        metadata: 环境特定的附加信息。

    设计说明:
        统一结构使规则控制器和强化学习策略能共享环境接口；数据在构造时复制，
        防止策略随后修改原数组而改变已经提交给环境的动作。
    """

    agent_id: Hashable
    action_type: ActionType
    value: np.ndarray | int | None
    timestamp: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.agent_id is None:
            raise ValueError("agent_id 不能为 None")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("动作时间必须是有限数值")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.action_type is ActionType.CONTINUOUS:
            vector = np.asarray(self.value, dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError("连续动作必须是一维非空向量")
            if not np.all(np.isfinite(vector)):
                raise ValueError("连续动作不能包含 NaN 或无穷值")
            object.__setattr__(self, "value", vector.copy())
        elif self.action_type is ActionType.DISCRETE:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, np.integer)):
                raise ValueError("离散动作必须使用整数编号")
            object.__setattr__(self, "value", int(self.value))
        elif self.action_type is ActionType.NOOP:
            if self.value is not None:
                raise ValueError("空操作的 value 必须为 None")
        else:
            raise ValueError(f"不支持的动作类型: {self.action_type!r}")

    @classmethod
    def continuous(
        cls,
        agent_id: Hashable,
        values: Sequence[float] | np.ndarray,
        *,
        timestamp: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Action":
        """构造连续动作。"""

        return cls(
            agent_id=agent_id,
            action_type=ActionType.CONTINUOUS,
            value=np.asarray(values, dtype=np.float64),
            timestamp=timestamp,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def discrete(
        cls,
        agent_id: Hashable,
        value: int,
        *,
        timestamp: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Action":
        """构造离散动作。"""

        return cls(
            agent_id=agent_id,
            action_type=ActionType.DISCRETE,
            value=value,
            timestamp=timestamp,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def noop(cls, agent_id: Hashable, *, timestamp: float = 0.0) -> "Action":
        """构造保持当前状态的空操作。"""

        return cls(agent_id, ActionType.NOOP, None, timestamp)
