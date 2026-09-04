"""智能体属性、动态状态和统一抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Hashable, Mapping, Protocol, runtime_checkable

import numpy as np

from .action import Action
from .message import Message
from .observation import Observation


@runtime_checkable
class AgentEnvironment(Protocol):
    """智能体依赖的最小环境协议，避免与具体环境实现循环耦合。"""

    @property
    def current_time(self) -> float: ...

    def route_message(self, message: Message) -> Any: ...

    def apply_action(self, agent_id: Hashable, action: Action) -> Any: ...


@dataclass(frozen=True)
class AgentAttributes:
    """智能体生命周期内通常不变的能力参数。"""

    agent_type: str = "base"
    max_speed: float = 1.0
    sensor_range: float = 10.0
    communication_range: float = 10.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("max_speed", "sensor_range", "communication_range"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正有限数值")
            object.__setattr__(self, name, value)
        if not self.agent_type:
            raise ValueError("agent_type 不能为空")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass
class AgentState:
    """智能体随仿真推进而变化的物理与运行状态。

    使用 ``default_factory`` 为每个智能体创建独立数组，避免多个实例共享位置或
    速度对象。拓扑邻居不直接保存在这里，而由 ``Observation`` 按当前版本提供，
    防止拓扑更新后留下过期邻居引用。
    """

    position: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    yaw: float = 0.0
    energy: float = 1.0
    active: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = self._validated_vector(self.position, "position")
        self.velocity = self._validated_vector(self.velocity, "velocity")
        if self.position.size != self.velocity.size:
            raise ValueError("position 与 velocity 维度必须相同")
        self.yaw = float(self.yaw)
        self.energy = float(self.energy)
        if not math.isfinite(self.yaw):
            raise ValueError("yaw 必须是有限数值")
        if not math.isfinite(self.energy) or self.energy < 0:
            raise ValueError("energy 必须是非负有限数值")
        self.extras = dict(self.extras)

    def as_vector(self) -> np.ndarray:
        """返回基础物理状态向量，供观测构造和控制算法使用。"""

        return np.concatenate(
            (self.position, self.velocity, np.asarray([self.yaw, self.energy]))
        )

    @property
    def pos(self) -> np.ndarray:
        """兼容早期架构文档的 ``pos`` 属性名。"""

        return self.position

    @pos.setter
    def pos(self, value: np.ndarray) -> None:
        self.position = self._validated_vector(value, "pos")

    @property
    def vel(self) -> np.ndarray:
        """兼容早期架构文档的 ``vel`` 属性名。"""

        return self.velocity

    @vel.setter
    def vel(self, value: np.ndarray) -> None:
        self.velocity = self._validated_vector(value, "vel")

    @staticmethod
    def _validated_vector(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"{name} 必须是一维非空向量")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} 不能包含 NaN 或无穷值")
        return vector.copy()


class BaseAgent(ABC):
    """规则策略与学习策略共享的智能体基类。

    参数:
        agent_id: 全局唯一标识。
        attributes: 静态能力参数。
        state: 初始动态状态。

    子类只需实现 ``perceive`` 和 ``decide``。环境统一收集所有动作后再执行，
    避免按智能体遍历顺序立即更新状态造成不公平的顺序偏差。
    """

    def __init__(
        self,
        agent_id: Hashable,
        attributes: AgentAttributes | None = None,
        state: AgentState | None = None,
    ) -> None:
        if agent_id is None:
            raise ValueError("agent_id 不能为 None")
        self.id = agent_id
        self.attributes = attributes or AgentAttributes()
        self.attr = self.attributes
        self.state = state or AgentState()
        self._environment: AgentEnvironment | None = None
        self._last_observation: Observation | None = None
        self._inbox: list[Message] = []

    @property
    def environment(self) -> AgentEnvironment | None:
        """返回当前绑定环境。"""

        return self._environment

    @property
    def last_observation(self) -> Observation | None:
        """返回最近一次处理的观测。"""

        return self._last_observation

    @property
    def inbox_size(self) -> int:
        """返回尚未消费的消息数。"""

        return len(self._inbox)

    def bind_environment(self, environment: AgentEnvironment | None) -> None:
        """由环境建立或解除双向关联。"""

        if environment is not None and not isinstance(environment, AgentEnvironment):
            raise TypeError("environment 未实现 AgentEnvironment 协议")
        self._environment = environment

    def bind_env(self, environment: AgentEnvironment | None) -> None:
        """兼容架构原稿名称，行为与 ``bind_environment`` 相同。"""

        self.bind_environment(environment)

    @abstractmethod
    def perceive(self, observation: Observation) -> Any:
        """将环境观测转换为策略内部表示。"""

    @abstractmethod
    def decide(self, dt: float) -> Action:
        """根据内部表示生成当前时间步动作。"""

    def step(self, observation: Observation, dt: float) -> Action:
        """处理观测并生成动作，但不立即改变环境状态。"""

        if observation.agent_id != self.id:
            raise ValueError("观测所属智能体与当前智能体不一致")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt 必须是正有限数值")
        self._last_observation = observation
        self.perceive(observation)
        action = self.decide(float(dt))
        if action.agent_id != self.id:
            raise ValueError("策略返回了属于其他智能体的动作")
        return action

    def act(self, action: Action, dt: float | None = None) -> Any:
        """把动作提交给环境；通常由环境在统一收集动作后调用。"""

        if self._environment is None:
            raise RuntimeError("智能体尚未绑定环境")
        if action.agent_id != self.id:
            raise ValueError("不能执行其他智能体的动作")
        if dt is not None and (not math.isfinite(float(dt)) or dt <= 0):
            raise ValueError("dt 必须是正有限数值")
        return self._environment.apply_action(self.id, action)

    def send_message(
        self,
        receiver: Hashable | None,
        payload: Any,
        *,
        ttl: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """通过环境发送单播或邻居广播消息。"""

        if self._environment is None:
            raise RuntimeError("智能体尚未绑定环境")
        message = Message(
            sender=self.id,
            receiver=receiver,
            payload=payload,
            created_at=self._environment.current_time,
            ttl=ttl,
            metadata=dict(metadata or {}),
        )
        return self._environment.route_message(message)

    def receive_message(self, message: Message) -> None:
        """接收通信总线投递的消息。"""

        if message.receiver not in {None, self.id}:
            raise ValueError("消息接收方与当前智能体不一致")
        self._inbox.append(message)

    def send_msg(self, target: Hashable | None, content: Any) -> Any:
        """兼容架构原稿的简化发送接口。"""

        return self.send_message(target, content)

    def receive_msg(self, message: Message) -> None:
        """兼容架构原稿的简化接收接口。"""

        self.receive_message(message)

    def pop_messages(self) -> tuple[Message, ...]:
        """按到达顺序取出并清空消息队列。"""

        messages = tuple(self._inbox)
        self._inbox.clear()
        return messages
