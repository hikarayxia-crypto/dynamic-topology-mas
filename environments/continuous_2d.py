"""连续二维空间中的多智能体协同搜索环境。"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from core.action import Action, ActionType
from core.agent import BaseAgent
from core.environment import BaseEnvironment, StepResult
from core.message import Message
from core.observation import NeighborObservation, Observation
from core.topology import DynamicTopology
from interaction.communication import (
    CommunicationBus,
    CommunicationConfig,
    TransmissionResult,
)


@dataclass(frozen=True)
class Continuous2DConfig:
    """连续二维协同搜索环境参数。

    目标位置为 ``None`` 时每次重置随机生成；测试或可复现实验可以直接提供固定
    坐标。奖励采用全队共享形式，为后续参数共享策略提供一致协作信号。
    """

    width: float = 50.0
    height: float = 50.0
    dt: float = 0.2
    max_steps: int = 500
    n_targets: int = 8
    target_positions: tuple[tuple[float, float], ...] | None = None
    detection_reward: float = 10.0
    completion_bonus: float = 20.0
    step_penalty: float = 0.01
    collision_distance: float = 0.5
    collision_penalty: float = 1.0
    connectivity_penalty: float = 0.1
    energy_cost: float = 0.001
    replacement_lane_tolerance: float = 1.0

    def __post_init__(self) -> None:
        for name in ("width", "height", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正有限数值")
            object.__setattr__(self, name, value)
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")
        if self.n_targets < 0:
            raise ValueError("n_targets 不能为负数")
        for name in (
            "detection_reward",
            "completion_bonus",
            "step_penalty",
            "collision_distance",
            "collision_penalty",
            "connectivity_penalty",
            "energy_cost",
            "replacement_lane_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数值")
            object.__setattr__(self, name, value)
        if self.target_positions is not None:
            positions = tuple(tuple(map(float, position)) for position in self.target_positions)
            if any(len(position) != 2 for position in positions):
                raise ValueError("每个目标位置必须包含两个坐标")
            if any(
                not 0 <= position[0] <= self.width
                or not 0 <= position[1] <= self.height
                for position in positions
            ):
                raise ValueError("目标位置必须位于环境边界内")
            object.__setattr__(self, "target_positions", positions)
            object.__setattr__(self, "n_targets", len(positions))


@dataclass
class SearchTarget:
    """环境中的未知搜索目标。"""

    target_id: int
    position: np.ndarray
    discovered: bool = False
    discovered_by: Hashable | None = None
    discovered_at: float | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("目标位置必须是二维有限向量")
        self.position = position.copy()


@dataclass(frozen=True)
class LinkFault:
    """指定时间区间内强制断开的通信链路。"""

    source: Hashable
    target: Hashable
    start_time: float
    end_time: float


@dataclass(frozen=True)
class NodeFault:
    """指定时间区间内强制离线的智能体。"""

    agent_id: Hashable
    start_time: float
    end_time: float


class Continuous2DSearchEnv(BaseEnvironment):
    """连续二维协同搜索环境。

    动作是二维归一化期望速度，环境根据智能体 ``max_speed`` 转换为实际速度。
    目标位置不会写入观测，智能体只能通过移动进入感知半径完成发现，避免规则
    基线或学习策略直接获得未知目标坐标。
    """

    SELF_FEATURE_DIM = 6
    NEIGHBOR_FEATURE_DIM = 5
    TASK_FEATURE_DIM = 2

    def __init__(
        self,
        config: Continuous2DConfig | None = None,
        communication_config: CommunicationConfig | None = None,
    ) -> None:
        super().__init__(DynamicTopology())
        self.config = config or Continuous2DConfig()
        self.communication_config = communication_config or CommunicationConfig()
        self.communication = CommunicationBus(
            self.topology, self.communication_config
        )
        self._rng = np.random.default_rng()
        self._targets: list[SearchTarget] = []
        self._step_count = 0
        self._episode_finished = False
        self._link_faults: list[LinkFault] = []
        self._node_faults: list[NodeFault] = []
        self._fault_forced_offline: set[Hashable] = set()
        self._last_delivery_results: tuple[TransmissionResult, ...] = ()
        self._last_info: dict[str, Any] = {}
        self._replacement_records: dict[str, dict[str, Any]] = {}
        self._replacement_known_missing: tuple[str, ...] = ()
        self._replacement_switches = 0
        self._replacement_step_coverage: dict[str, bool] = {}
        self._last_replacement_targets: tuple[str, ...] = ()
        self._replacement_roster: dict[str, Hashable] = {}
        self._replacement_roster_count = 0

    @property
    def step_count(self) -> int:
        """返回当前回合已执行时间步数。"""

        return self._step_count

    @property
    def targets(self) -> tuple[SearchTarget, ...]:
        """返回目标对象的只读容器；目标状态仍由环境管理。"""

        return tuple(self._targets)

    @property
    def detection_ratio(self) -> float:
        """返回已发现目标比例，无目标任务按全部完成处理。"""

        if not self._targets:
            return 1.0
        return sum(target.discovered for target in self._targets) / len(self._targets)

    def add_agent(
        self,
        agent: BaseAgent,
        *,
        node_id: Hashable | None = None,
        node_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """添加智能体，并向通信总线登记端点—节点映射。"""

        super().add_agent(agent, node_id=node_id, node_metadata=node_metadata)
        self.communication.register_endpoint(agent.id, self.node_for_agent(agent.id))

    def remove_agent(self, agent_id: Hashable) -> BaseAgent:
        """移除智能体并清理通信端点。"""

        self.communication.unregister_endpoint(agent_id)
        return super().remove_agent(agent_id)

    def schedule_link_fault(
        self,
        source_id: Hashable,
        target_id: Hashable,
        *,
        start_time: float,
        duration: float,
    ) -> None:
        """安排链路故障；到期后若距离允许会自动重连。"""

        self.get_agent(source_id)
        self.get_agent(target_id)
        start, end = self._fault_interval(start_time, duration)
        self._link_faults.append(LinkFault(source_id, target_id, start, end))

    def schedule_node_fault(
        self,
        agent_id: Hashable,
        *,
        start_time: float,
        duration: float,
    ) -> None:
        """安排节点离线与自动恢复。"""

        self.get_agent(agent_id)
        start, end = self._fault_interval(start_time, duration)
        self._node_faults.append(NodeFault(agent_id, start, end))

    def reset(self, seed: int | None = None) -> Mapping[Hashable, Observation]:
        """重置位置、能量、目标、拓扑、通信队列和回合统计。"""

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_finished = False
        self._fault_forced_offline.clear()
        self._last_delivery_results = ()
        self._last_info = {}
        # 补位响应和恢复时间属于单回合实验状态，重置时不得泄漏到下一回合。
        self._replacement_records.clear()
        self._replacement_known_missing = ()
        self._replacement_switches = 0
        self._replacement_step_coverage = {}
        self._last_replacement_targets = ()
        # 每次 reset 固化本回合成员表；中途增删节点不能改变任务身份和指标分母。
        self._replacement_roster = {
            str(agent_id): agent_id for agent_id in self._agents
        }
        self._replacement_roster_count = len(self._replacement_roster)

        # 每回合创建新拓扑，避免历史版本、断边缓存和待投递消息泄漏到下一回合。
        self.topology = DynamicTopology()
        self.communication = CommunicationBus(
            self.topology, self.communication_config, seed=seed
        )
        for agent_id, agent in self._agents.items():
            node_id = self._agent_nodes[agent_id]
            self.topology.add_node(node_id, {"agent_id": agent_id})
            self.communication.register_endpoint(agent_id, node_id)
            agent.state.position = self._rng.uniform(
                [0.0, 0.0], [self.config.width, self.config.height]
            )
            agent.state.velocity = np.zeros(2, dtype=np.float64)
            agent.state.yaw = 0.0
            agent.state.energy = 1.0
            agent.state.active = True
            agent.state.extras["discoveries"] = 0
            agent.pop_messages()

        self._targets = self._create_targets()
        self._apply_node_faults(self.current_time)
        self._refresh_distance_topology(self.current_time)
        return self._build_observations()

    def step(self, actions: Mapping[Hashable, Action]) -> StepResult:
        """同步执行动作、推进故障与通信、探测目标并计算共享奖励。"""

        if self._episode_finished:
            raise RuntimeError("回合已经结束，请先调用 reset()")
        self.validate_action_batch(actions)

        for agent_id, action in actions.items():
            self.apply_action(agent_id, action)

        next_time = self.current_time + self.config.dt
        # 动作已经通过批校验并执行，应先按真实新位置记账；步末刚发生的故障只影响
        # 下一步动作资格，不能抹掉本步已经完成的补位控制。
        self._update_replacement_tracking(actions, next_time)
        self.topology.advance_time(next_time)
        self._step_count += 1
        self._apply_node_faults(next_time)
        self._refresh_distance_topology(next_time)
        self._last_delivery_results = self.communication.advance_time(
            next_time, self._agents
        )

        newly_discovered = self._detect_targets(next_time)
        collision_count = self._count_collisions()
        connectivity = self._connectivity_ratio()
        success = all(target.discovered for target in self._targets)
        no_active_agents = not any(agent.state.active for agent in self._agents.values())
        reached_limit = self._step_count >= self.config.max_steps
        truncated = (reached_limit or no_active_agents) and not success
        self._episode_finished = success or truncated

        team_reward = (
            newly_discovered * self.config.detection_reward
            - self.config.step_penalty
            - collision_count * self.config.collision_penalty
            - (1.0 - connectivity) * self.config.connectivity_penalty
        )
        if success:
            team_reward += self.config.completion_bonus

        rewards = {
            agent_id: team_reward if agent.state.active else 0.0
            for agent_id, agent in self._agents.items()
        }
        terminated_map = {agent_id: success for agent_id in self._agents}
        truncated_map = {agent_id: truncated for agent_id in self._agents}
        observations = self._build_observations()
        info = {
            "time": next_time,
            "step": self._step_count,
            "targets_total": len(self._targets),
            "targets_discovered": sum(target.discovered for target in self._targets),
            "newly_discovered": newly_discovered,
            "detection_ratio": self.detection_ratio,
            "collision_count": collision_count,
            "connectivity_ratio": connectivity,
            "topology_version": self.topology.version,
            "messages_delivered": sum(
                result.status.value == "delivered"
                for result in self._last_delivery_results
            ),
            "success": success,
            "truncated": truncated,
            "no_active_agents": no_active_agents,
            "known_missing": self._replacement_known_missing,
            "replacement_active_count": len(self._last_replacement_targets),
            "replacement_targets": self._last_replacement_targets,
            "replacement_coverage_restored": dict(self._replacement_step_coverage),
            "uncovered_lane_ratio": (
                sum(not restored for restored in self._replacement_step_coverage.values())
                / max(1, self._replacement_roster_count)
            ),
            "replacement_switches": self._replacement_switches,
        }
        self._last_info = info
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated_map,
            truncated=truncated_map,
            info=info,
        )

    def route_message(self, message: Message) -> tuple[TransmissionResult, ...]:
        """通过当前动态拓扑发送消息。"""

        return self.communication.send(message, current_time=self.current_time)

    def apply_action(self, agent_id: Hashable, action: Action) -> None:
        """应用二维归一化速度动作，并处理边界、朝向和能量消耗。"""

        agent = self.get_agent(agent_id)
        if action.agent_id != agent_id:
            raise ValueError("动作所属智能体与映射键不一致")
        if not agent.state.active:
            raise ValueError("离线智能体不能执行动作")

        if action.action_type is ActionType.NOOP:
            command = np.zeros(2, dtype=np.float64)
        elif action.action_type is ActionType.CONTINUOUS:
            command = np.asarray(action.value, dtype=np.float64)
            if command.shape != (2,):
                raise ValueError("连续二维环境要求动作向量形状为 (2,)")
        else:
            raise ValueError("连续二维环境暂不接受离散动作")

        norm = float(np.linalg.norm(command))
        if norm > 1.0:
            command = command / norm
        velocity = command * agent.attributes.max_speed
        new_position = agent.state.position + velocity * self.config.dt
        agent.state.position = np.clip(
            new_position,
            [0.0, 0.0],
            [self.config.width, self.config.height],
        )
        agent.state.velocity = velocity
        if np.linalg.norm(velocity) > 0:
            agent.state.yaw = math.atan2(velocity[1], velocity[0])
        agent.state.energy = max(
            0.0,
            agent.state.energy
            - self.config.energy_cost * float(np.linalg.norm(velocity)) * self.config.dt,
        )
        if agent.state.energy == 0.0:
            agent.state.active = False
            self.topology.set_node_active(self.node_for_agent(agent_id), False)

    def target_snapshot(self) -> tuple[dict[str, Any], ...]:
        """返回可记录但不会暴露给策略的目标状态快照。"""

        return tuple(
            {
                "target_id": target.target_id,
                "position": target.position.copy(),
                "discovered": target.discovered,
                "discovered_by": target.discovered_by,
                "discovered_at": target.discovered_at,
            }
            for target in self._targets
        )

    def replacement_snapshot(self) -> dict[str, Any]:
        """返回当前回合补位历史的副本，供评估读取而不暴露内部可变字典。"""

        return {
            "known_missing": tuple(self._replacement_known_missing),
            "tasks": {
                missing_id: dict(record)
                for missing_id, record in self._replacement_records.items()
            },
            "replacement_switches": self._replacement_switches,
        }

    def _update_replacement_tracking(
        self,
        actions: Mapping[Hashable, Action],
        timestamp: float,
    ) -> None:
        """解析实际提交动作，并用执行后的真实位置更新补位响应和覆盖状态。

        参数为本步动作批和对应结束时刻；没有返回值。策略声明只用于识别任务，
        是否恢复覆盖必须由环境中的真实纵坐标判定，防止自报结果污染实验指标。
        """

        roster = dict(self._replacement_roster)
        known_missing: set[str] = set()
        for action in actions.values():
            if action.action_type is not ActionType.CONTINUOUS:
                continue
            reported = action.metadata.get("known_missing", ())
            if isinstance(reported, (str, bytes)) or not isinstance(reported, Sequence):
                continue
            for missing_id in reported:
                missing_key = str(missing_id)
                if missing_key in roster:
                    known_missing.add(missing_key)

        candidates: dict[str, tuple[float, str, float]] = {}
        for agent_id, action in actions.items():
            if action.action_type is not ActionType.CONTINUOUS:
                continue
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            metadata = action.metadata
            if "replacement_for" not in metadata:
                continue
            missing_id = str(metadata.get("replacement_for"))
            replacement_agent = str(metadata.get("replacement_agent", agent_id))
            lane_y = metadata.get("replacement_lane_y")
            score = metadata.get("replacement_bid_score")
            if (
                str(agent_id) not in roster
                or missing_id not in roster
                or missing_id == str(agent_id)
                or missing_id not in known_missing
                or replacement_agent != str(agent_id)
                or isinstance(lane_y, bool)
                or not isinstance(lane_y, (int, float))
                or not math.isfinite(lane_y)
                or not 0.0 <= float(lane_y) <= self.config.height
                or (
                    score is not None
                    and (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(score)
                    )
                )
            ):
                continue
            ranking_score = math.inf if score is None else float(score)
            candidate = (ranking_score, str(agent_id), float(lane_y))
            previous = candidates.get(missing_id)
            if previous is None or candidate[:2] < previous[:2]:
                candidates[missing_id] = candidate

        self._replacement_known_missing = tuple(sorted(known_missing))
        self._last_replacement_targets = tuple(sorted(candidates))
        self._replacement_step_coverage = {
            missing_id: False for missing_id in self._replacement_known_missing
        }
        for record in self._replacement_records.values():
            record["coverage_restored"] = False

        for missing_id, (score, replacer_id, lane_y) in candidates.items():
            agent = self._agents[roster[replacer_id]]
            record = self._replacement_records.get(missing_id)
            if record is None:
                record = {
                    "current_replacer": replacer_id,
                    "lane_y": lane_y,
                    "bid_score": None if math.isinf(score) else score,
                    "first_response_at": timestamp,
                    "first_coverage_at": None,
                    "switches": 0,
                    "coverage_restored": False,
                }
                self._replacement_records[missing_id] = record
            else:
                if record["current_replacer"] != replacer_id:
                    record["switches"] += 1
                    self._replacement_switches += 1
                record["current_replacer"] = replacer_id
                record["lane_y"] = lane_y
                record["bid_score"] = None if math.isinf(score) else score

            tolerance = max(
                self.config.replacement_lane_tolerance,
                agent.attributes.sensor_range,
            )
            restored = abs(float(agent.state.position[1]) - lane_y) <= tolerance
            self._replacement_step_coverage[missing_id] = restored
            record["coverage_restored"] = restored
            if restored and record["first_coverage_at"] is None:
                record["first_coverage_at"] = timestamp

    def _create_targets(self) -> list[SearchTarget]:
        if self.config.target_positions is not None:
            positions: Sequence[Sequence[float]] = self.config.target_positions
        else:
            positions = self._rng.uniform(
                [0.0, 0.0],
                [self.config.width, self.config.height],
                size=(self.config.n_targets, 2),
            )
        return [
            SearchTarget(index, np.asarray(position, dtype=np.float64))
            for index, position in enumerate(positions)
        ]

    def _build_observations(self) -> dict[Hashable, Observation]:
        observations: dict[Hashable, Observation] = {}
        node_to_agent = {node: agent for agent, node in self._agent_nodes.items()}
        active_ids = [
            agent_id for agent_id, agent in self._agents.items() if agent.state.active
        ]
        for agent_index, agent_id in enumerate(active_ids):
            agent = self._agents[agent_id]
            node_id = self._agent_nodes[agent_id]
            self_features = np.asarray(
                [
                    agent.state.position[0] / self.config.width,
                    agent.state.position[1] / self.config.height,
                    agent.state.velocity[0] / agent.attributes.max_speed,
                    agent.state.velocity[1] / agent.attributes.max_speed,
                    agent.state.energy,
                    agent.state.yaw / math.pi,
                ],
                dtype=np.float64,
            )
            neighbors: list[NeighborObservation] = []
            for neighbor_node in self.topology.get_neighbors(node_id):
                neighbor_id = node_to_agent.get(neighbor_node)
                if neighbor_id is None:
                    continue
                neighbor = self._agents[neighbor_id]
                relative_position = neighbor.state.position - agent.state.position
                relative_velocity = neighbor.state.velocity - agent.state.velocity
                distance = float(np.linalg.norm(relative_position))
                neighbors.append(
                    NeighborObservation(
                        neighbor_id=neighbor_id,
                        features=np.asarray(
                            [
                                relative_position[0] / self.config.width,
                                relative_position[1] / self.config.height,
                                relative_velocity[0] / agent.attributes.max_speed,
                                relative_velocity[1] / agent.attributes.max_speed,
                                neighbor.state.energy,
                            ]
                        ),
                        link_weight=1.0 / (1.0 + distance),
                        metadata={
                            "distance": distance,
                            "relative_position": relative_position.copy(),
                        },
                    )
                )
            observations[agent_id] = Observation(
                agent_id=agent_id,
                self_features=self_features,
                neighbors=tuple(neighbors),
                task_features=np.asarray(
                    [
                        self.detection_ratio,
                        max(0.0, 1.0 - self._step_count / self.config.max_steps),
                    ]
                ),
                topology_version=self.topology.version,
                timestamp=self.current_time,
                neighbor_feature_dim=self.NEIGHBOR_FEATURE_DIM,
                metadata={
                    "world_size": (self.config.width, self.config.height),
                    "position": agent.state.position.copy(),
                    "agent_index": agent_index,
                    "agent_count": len(active_ids),
                    "sensor_range": agent.attributes.sensor_range,
                    "all_targets_discovered": self.detection_ratio == 1.0,
                    "discoveries": agent.state.extras.get("discoveries", 0),
                },
            )
        return observations

    def _detect_targets(self, timestamp: float) -> int:
        newly_discovered = 0
        active_agents = [
            agent for agent in self._agents.values() if agent.state.active
        ]
        for target in self._targets:
            if target.discovered:
                continue
            candidates = [
                (
                    float(np.linalg.norm(agent.state.position - target.position)),
                    agent,
                )
                for agent in active_agents
                if np.linalg.norm(agent.state.position - target.position)
                <= agent.attributes.sensor_range
            ]
            if not candidates:
                continue
            _, discoverer = min(candidates, key=lambda item: item[0])
            target.discovered = True
            target.discovered_by = discoverer.id
            target.discovered_at = timestamp
            discoverer.state.extras["discoveries"] = (
                discoverer.state.extras.get("discoveries", 0) + 1
            )
            newly_discovered += 1
        return newly_discovered

    def _refresh_distance_topology(self, timestamp: float) -> None:
        for first_id, second_id in combinations(self._agents, 2):
            first = self._agents[first_id]
            second = self._agents[second_id]
            first_node = self._agent_nodes[first_id]
            second_node = self._agent_nodes[second_id]
            if not first.state.active or not second.state.active:
                continue
            distance = float(np.linalg.norm(first.state.position - second.state.position))
            communication_range = min(
                first.attributes.communication_range,
                second.attributes.communication_range,
            )
            should_connect = (
                distance <= communication_range
                and not self._is_link_faulted(first_id, second_id, timestamp)
            )
            connected = self.topology.are_directly_connected(
                first_node, second_node, active_only=False
            )
            if should_connect and not connected:
                self.topology.connect(first_node, second_node)
            elif not should_connect and connected:
                self.topology.disconnect(first_node, second_node)

    def _apply_node_faults(self, timestamp: float) -> None:
        for agent_id, agent in self._agents.items():
            fault_active = any(
                fault.agent_id == agent_id
                and fault.start_time <= timestamp < fault.end_time
                for fault in self._node_faults
            )
            if fault_active and agent_id not in self._fault_forced_offline:
                if agent.state.active:
                    agent.state.active = False
                    self.topology.set_node_active(self._agent_nodes[agent_id], False)
                self._fault_forced_offline.add(agent_id)
            elif not fault_active and agent_id in self._fault_forced_offline:
                self._fault_forced_offline.remove(agent_id)
                if agent.state.energy > 0:
                    agent.state.active = True
                    self.topology.set_node_active(self._agent_nodes[agent_id], True)

    def _is_link_faulted(
        self, first_id: Hashable, second_id: Hashable, timestamp: float
    ) -> bool:
        endpoints = frozenset((first_id, second_id))
        return any(
            frozenset((fault.source, fault.target)) == endpoints
            and fault.start_time <= timestamp < fault.end_time
            for fault in self._link_faults
        )

    def _count_collisions(self) -> int:
        active_positions = [
            agent.state.position
            for agent in self._agents.values()
            if agent.state.active
        ]
        return int(
            sum(
                np.linalg.norm(first - second) < self.config.collision_distance
                for first, second in combinations(active_positions, 2)
            )
        )

    def _connectivity_ratio(self) -> float:
        active_nodes = self.topology.nodes(active_only=True)
        pairs = list(combinations(active_nodes, 2))
        if not pairs:
            return 1.0
        connected_pairs = sum(
            self.topology.is_reachable(first, second) for first, second in pairs
        )
        return connected_pairs / len(pairs)

    def _fault_interval(self, start_time: float, duration: float) -> tuple[float, float]:
        start = float(start_time)
        length = float(duration)
        if not math.isfinite(start) or start < self.current_time:
            raise ValueError("故障开始时间不能早于当前仿真时间")
        if not math.isfinite(length) or length <= 0:
            raise ValueError("故障持续时间必须是正有限数值")
        return start, start + length
