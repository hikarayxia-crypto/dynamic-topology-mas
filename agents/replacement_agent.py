"""基于本地 gossip 协调结果的可恢复搜索带补位智能体。"""

from __future__ import annotations

from typing import Hashable

from core.action import Action
from core.agent import AgentAttributes, AgentState
from core.observation import Observation
from coordination.replacement import (
    ReplacementAssignment,
    ReplacementConfig,
    ReplacementCoordinator,
)

from .rule_based_agent import RuleBasedSearchAgent


class ReplacementSearchAgent(RuleBasedSearchAgent):
    """在规则扫描上叠加去中心化竞价、广播和稳定驻留的补位控制。

    参数包括自身标识、固定成员表和世界高度；``replacement_config`` 控制故障
    确认、竞价和驻留窗口，其余参数保持规则扫描智能体的构造兼容性。所有赢家均
    由本地协调器按 gossip 复现，环境既不读取目标坐标也不决定赢家。
    """

    def __init__(
        self,
        agent_id: Hashable,
        roster: tuple[Hashable, ...] | list[Hashable],
        world_height: float,
        attributes: AgentAttributes | None = None,
        state: AgentState | None = None,
        *,
        replacement_config: ReplacementConfig | None = None,
        separation_distance: float = 1.5,
        separation_strength: float = 1.2,
    ) -> None:
        super().__init__(
            agent_id,
            attributes,
            state,
            separation_distance=separation_distance,
            separation_strength=separation_strength,
        )
        self._coordinator = ReplacementCoordinator(
            agent_id, roster, world_height, replacement_config or ReplacementConfig()
        )
        self._last_broadcast_at: float | None = None
        # 缓存当前分配直到驻留周期结束，避免恢复事件使控制指令在周期中跳变。
        self._resident_assignment: ReplacementAssignment | None = None
        self._resident_steps = 0
        self._last_completed_missing_id: str | None = None

    @property
    def coordinator(self) -> ReplacementCoordinator:
        """返回只供检查的协调器引用；分配仍只能经协调协议更新。"""

        return self._coordinator

    def perceive(self, observation: Observation) -> Observation:
        """消费本步协议消息并同步本地竞价输入。

        参数为本机观测；返回同一观测以保持规则智能体接口。仅接收
        ``replacement_gossip`` 负载，损坏或非协议消息由协调器拒绝后忽略，避免
        通信故障中止控制循环。
        """

        super().perceive(observation)
        self._coordinator.advance_time(observation.timestamp)
        for message in self.pop_messages():
            payload = message.payload
            if isinstance(payload, dict) and payload.get("kind") == "replacement_gossip":
                self._coordinator.ingest_gossip(payload, observation.timestamp)
        position_y = float(observation.metadata["position"][1])
        self._coordinator.update_local_status(
            position_y, self.state.energy, len(observation.neighbors), observation.timestamp
        )
        return observation

    def decide(self, dt: float) -> Action:
        """广播本地共识、竞价缺失搜索带并输出驻留后的扫描动作。

        ``dt`` 由规则扫描用于边界折返；广播仅使用观测时间，避免墙钟导致仿真
        不可复现。返回规则动作或附带补位元数据的扫描动作。
        """

        observation = self._observation
        if observation is None or not self.state.active:
            return Action.noop(self.id, timestamp=self._timestamp())
        if observation.metadata.get("all_targets_discovered", False):
            return Action.noop(self.id, timestamp=observation.timestamp)

        now = observation.timestamp
        self._bid_for_missing_nodes(now)
        self._broadcast_if_due(now)
        assignment = self._next_resident_assignment()
        if assignment is None:
            base_action = super().decide(dt)
            # 即使本步没有执行补位，也要报告本地已确认缺失集合，环境才能区分
            # “尚无补位者”与“根本没有缺失任务”，但不暴露任何目标真实坐标。
            return Action.continuous(
                self.id,
                base_action.value,
                timestamp=base_action.timestamp,
                metadata={
                    **base_action.metadata,
                    "known_missing": self._known_missing_for_tracking(),
                },
            )

        command = self._sweep_command(observation, dt, target_y=assignment.lane.center_y)
        bid = self._coordinator.known_bids.get(str(assignment.missing_id), {}).get(str(self.id))
        return Action.continuous(
            self.id,
            command,
            timestamp=observation.timestamp,
            metadata={
                "policy": "replacement_sweep",
                "known_missing": self._known_missing_for_tracking(),
                "replacement_for": str(assignment.missing_id),
                "replacement_agent": str(self.id),
                "replacement_lane_y": float(assignment.lane.center_y),
                "replacement_bid_score": None if bid is None else float(bid.score),
            },
        )

    def _known_missing_for_tracking(self) -> tuple[str, ...]:
        """返回环境跟踪所需的稳定缺失集合，不把协调器内部可变对象外泄。"""

        # RECOVERING 阶段仍保留旧分配并继续覆盖，因此其任务也应报告为尚未交还。
        task_ids = set(self._coordinator.missing_nodes)
        task_ids.update(str(missing_id) for missing_id in self._coordinator.assignments)
        return tuple(sorted(task_ids))

    def _bid_for_missing_nodes(self, timestamp: float) -> None:
        """为每个已确认缺失节点登记本地竞价，单项输入错误不会阻断动作。"""

        for missing_id in self._coordinator.missing_nodes:
            try:
                # 前一项可能已在零窗口内完成分配，下一项必须据此增加负载成本。
                current_load = sum(
                    assignment.winner_id == str(self.id)
                    for assignment in self._coordinator.assignments.values()
                )
                self._coordinator.local_bid_for(missing_id, current_load, timestamp)
            except (KeyError, ValueError):
                # 协调器把竞价前置条件作为输入错误；网络瞬态下跳过该项即可。
                continue

    def _broadcast_if_due(self, timestamp: float) -> None:
        """按仿真时间间隔向当前邻居广播，未绑定或拒绝发送不影响控制。"""

        if (
            self._last_broadcast_at is not None
            and timestamp - self._last_broadcast_at < self._coordinator.config.broadcast_interval
        ):
            return
        try:
            self.send_message(None, self._coordinator.build_gossip())
        except RuntimeError:
            # 未绑定环境或传输层拒绝均是可预期的分区情形，仍保留本地控制。
            pass
        self._last_broadcast_at = timestamp

    def _next_resident_assignment(self) -> ReplacementAssignment | None:
        """选择本轮补位任务，并以缓存完成整个驻留周期后再平滑交还。"""

        if self._resident_assignment is not None and self._resident_steps < self._coordinator.config.dwell_steps:
            self._resident_steps += 1
            assignment = self._resident_assignment
            if self._resident_steps == self._coordinator.config.dwell_steps:
                self._last_completed_missing_id = str(assignment.missing_id)
            return assignment

        self._resident_assignment = None
        self._resident_steps = 0
        assignments = sorted(
            (
                assignment
                for assignment in self._coordinator.assignments.values()
                if assignment.winner_id == str(self.id)
            ),
            key=lambda assignment: (assignment.assigned_at, str(assignment.missing_id)),
        )
        if not assignments:
            self._last_completed_missing_id = None
            return None
        start_index = 0
        if self._last_completed_missing_id is not None:
            for index, assignment in enumerate(assignments):
                if str(assignment.missing_id) == self._last_completed_missing_id:
                    start_index = (index + 1) % len(assignments)
                    break
        self._resident_assignment = assignments[start_index]
        self._resident_steps = 1
        if self._coordinator.config.dwell_steps == 1:
            self._last_completed_missing_id = str(self._resident_assignment.missing_id)
        return self._resident_assignment
