"""多智能体仿真环境的抽象生命周期与公共管理逻辑。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping

from .action import Action
from .agent import BaseAgent
from .message import Message
from .observation import Observation
from .topology import DynamicTopology, TopologyError


@dataclass(frozen=True)
class StepResult:
    """一个环境时间步的标准返回结果。"""

    observations: Mapping[Hashable, Observation]
    rewards: Mapping[Hashable, float]
    terminated: Mapping[Hashable, bool]
    truncated: Mapping[Hashable, bool]
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", dict(self.observations))
        object.__setattr__(self, "rewards", dict(self.rewards))
        object.__setattr__(self, "terminated", dict(self.terminated))
        object.__setattr__(self, "truncated", dict(self.truncated))
        object.__setattr__(self, "info", dict(self.info))


class BaseEnvironment(ABC):
    """具体网格或连续环境需要继承的抽象基类。

    基类负责智能体与拓扑节点的生命周期一致性，具体环境负责生成观测、计算
    奖励和更新物理状态。这样可以保证所有任务共享同一套动态图语义。
    """

    def __init__(self, topology: DynamicTopology | None = None) -> None:
        self.topology = topology or DynamicTopology()
        self._agents: dict[Hashable, BaseAgent] = {}
        self._agent_nodes: dict[Hashable, Hashable] = {}

    @property
    def current_time(self) -> float:
        """返回拓扑维护的统一仿真时间。"""

        return self.topology.current_time

    @property
    def agents(self) -> Mapping[Hashable, BaseAgent]:
        """返回只读智能体映射，防止外部绕过生命周期接口修改。"""

        return MappingProxyType(self._agents)

    def add_agent(
        self,
        agent: BaseAgent,
        *,
        node_id: Hashable | None = None,
        node_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """添加智能体并创建一一对应的拓扑节点。"""

        if agent.id in self._agents:
            raise ValueError(f"智能体已存在: {agent.id!r}")
        resolved_node = agent.id if node_id is None else node_id
        if resolved_node in self._agent_nodes.values():
            raise ValueError(f"拓扑节点已被其他智能体占用: {resolved_node!r}")
        if self.topology.has_node(resolved_node):
            raise ValueError(f"拓扑节点已存在且不受当前环境管理: {resolved_node!r}")
        self.topology.add_node(resolved_node, node_metadata)
        self._agents[agent.id] = agent
        self._agent_nodes[agent.id] = resolved_node
        agent.bind_environment(self)

    def remove_agent(self, agent_id: Hashable) -> BaseAgent:
        """移除智能体和对应拓扑节点，并返回被移除对象。"""

        if agent_id not in self._agents:
            raise KeyError(f"智能体不存在: {agent_id!r}")
        agent = self._agents.pop(agent_id)
        node_id = self._agent_nodes.pop(agent_id)
        self.topology.remove_node(node_id)
        agent.bind_environment(None)
        return agent

    def node_for_agent(self, agent_id: Hashable) -> Hashable:
        """查询智能体绑定的拓扑节点。"""

        try:
            return self._agent_nodes[agent_id]
        except KeyError as exc:
            raise KeyError(f"智能体不存在: {agent_id!r}") from exc

    def set_agent_active(self, agent_id: Hashable, active: bool) -> bool:
        """同步设置智能体运行状态与拓扑在线状态。"""

        agent = self.get_agent(agent_id)
        changed = self.topology.set_node_active(
            self.node_for_agent(agent_id), active
        )
        agent.state.active = active
        return changed

    def connect_agents(
        self, source_id: Hashable, target_id: Hashable, weight: float = 1.0
    ) -> bool:
        """使用智能体标识建立通信边。"""

        return self.topology.connect(
            self.node_for_agent(source_id),
            self.node_for_agent(target_id),
            weight=weight,
        )

    def get_agent(self, agent_id: Hashable) -> BaseAgent:
        """返回智能体，不存在时抛出 KeyError。"""

        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"智能体不存在: {agent_id!r}") from exc

    @abstractmethod
    def reset(self, seed: int | None = None) -> Mapping[Hashable, Observation]:
        """重置环境并返回每个在线智能体的初始观测。"""

    @abstractmethod
    def step(self, actions: Mapping[Hashable, Action]) -> StepResult:
        """同步执行一组动作并推进一个环境时间步。"""

    @abstractmethod
    def route_message(self, message: Message) -> Any:
        """把消息交给环境使用的通信实现。"""

    def route_msg(
        self, sender: Hashable, receiver: Hashable | None, content: Any
    ) -> Any:
        """兼容架构原稿接口，并统一转换为带时间戳的 ``Message``。"""

        return self.route_message(
            Message(
                sender=sender,
                receiver=receiver,
                payload=content,
                created_at=self.current_time,
            )
        )

    @abstractmethod
    def apply_action(self, agent_id: Hashable, action: Action) -> Any:
        """执行单个动作；具体物理约束由子类决定。"""

    def validate_action_batch(self, actions: Mapping[Hashable, Action]) -> None:
        """检查动作批次是否覆盖全部在线智能体且归属正确。"""

        expected = {
            agent_id for agent_id, agent in self._agents.items() if agent.state.active
        }
        provided = set(actions)
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            raise ValueError(f"动作批次不完整，缺少={missing!r}，多余={extra!r}")
        for agent_id, action in actions.items():
            if action.agent_id != agent_id:
                raise ValueError(f"动作映射键与动作归属不一致: {agent_id!r}")
