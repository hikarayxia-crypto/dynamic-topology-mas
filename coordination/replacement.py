"""补位协调所需的不可变配置和数据对象。"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence


def _is_finite_number(value: object) -> bool:
    """返回可安全表示为有限数值的 int 或 float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        # 超大整数无法安全转换为浮点数，按非法数值处理以维持 ValueError 契约。
        return False


@dataclass(frozen=True)
class ReplacementConfig:
    failure_timeout: float = 1.0
    failure_confirmation: float = 0.6
    bid_window: float = 0.4
    recovery_stability: float = 1.0
    broadcast_interval: float = 0.2
    bid_ttl: float = 2.0
    switch_margin: float = 0.05
    lane_tolerance: float = 1.0
    dwell_steps: int = 20
    distance_weight: float = 0.55
    load_weight: float = 0.20
    connectivity_weight: float = 0.15
    energy_weight: float = 0.10

    def __post_init__(self) -> None:
        # 时间、容差与评分参数必须为有限的非负数，避免协调逻辑出现无界状态。
        non_negative_fields = (
            "failure_timeout",
            "failure_confirmation",
            "bid_window",
            "recovery_stability",
            "broadcast_interval",
            "bid_ttl",
            "switch_margin",
            "lane_tolerance",
            "distance_weight",
            "load_weight",
            "connectivity_weight",
            "energy_weight",
        )
        for field_name in non_negative_fields:
            value = getattr(self, field_name)
            if not _is_finite_number(value) or value < 0:
                raise ValueError(f"{field_name} must be a finite non-negative number")

        if isinstance(self.dwell_steps, bool) or not isinstance(self.dwell_steps, int) or self.dwell_steps <= 0:
            raise ValueError("dwell_steps must be a positive integer")

        weight_total = (
            self.distance_weight
            + self.load_weight
            + self.connectivity_weight
            + self.energy_weight
        )
        if abs(weight_total - 1.0) > 1e-9:
            raise ValueError("replacement weights must sum to one")


@dataclass(frozen=True)
class CoverageLane:
    lane_id: str
    owner_id: str
    center_y: float


@dataclass(frozen=True)
class ReplacementBid:
    missing_id: Hashable
    bidder_id: Hashable
    score: float
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class ReplacementAssignment:
    missing_id: Hashable
    winner_id: Hashable
    assigned_at: float
    lane: CoverageLane


class NodeLiveness(str, Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    MISSING = "missing"
    RECOVERING = "recovering"


class ReplacementCoordinator:
    """维护成员心跳，并以确认期和恢复稳定期驱动存活状态。

    参数：agent_id 为本机标识，roster 为全体成员，world_height 用于验证固定搜索带，
    config 为协调超时配置。返回值：状态通过查询方法和 gossip 消息向外暴露。
    """

    def __init__(
        self,
        agent_id: Hashable,
        roster: Sequence[Hashable],
        world_height: float,
        config: ReplacementConfig,
    ) -> None:
        """以规范化成员表初始化心跳表和全部节点的健康状态。"""
        self.agent_id = str(agent_id)
        # 搜索带构建同时验证世界高度，并将成员标识统一为字符串。
        self.lanes = build_coverage_lanes(roster, world_height)
        self.roster = tuple(lane.owner_id for lane in self.lanes)
        if self.agent_id not in self.roster:
            raise ValueError("agent_id must be included in roster")
        if not isinstance(config, ReplacementConfig):
            raise TypeError("config must be a ReplacementConfig")

        self.config = config
        self.world_height = float(world_height)
        self._last_heartbeats: dict[str, float] = {node_id: 0.0 for node_id in self.roster}
        self._suspected_since: dict[str, float] = {}
        self._recovering_since: dict[str, float] = {}
        self._liveness: dict[str, NodeLiveness] = {
            node_id: NodeLiveness.HEALTHY for node_id in self.roster
        }
        self._last_time = 0.0
        self._local_status: tuple[float, float, int] | None = None
        self._known_bids: dict[str, dict[str, ReplacementBid]] = {}
        self._assignments: dict[str, ReplacementAssignment] = {}
        self._last_winners: dict[str, str] = {}
        self._bid_window_started: dict[str, float] = {}
        self._assignment_switch_count = 0

    @property
    def assignments(self) -> dict[str, ReplacementAssignment]:
        """返回缺失节点到当前补位分配的副本，供环境读取而不暴露内部可变状态。"""
        return dict(self._assignments)

    @property
    def known_bids(self) -> dict[str, dict[str, ReplacementBid]]:
        """返回按缺失节点和竞价者索引的未过期竞价副本，避免外部修改共识状态。"""
        return {missing_id: dict(bids) for missing_id, bids in self._known_bids.items()}

    @property
    def assignment_switch_count(self) -> int:
        """返回首次分配之外的赢家实际切换次数，供评估稳定性时使用。"""
        return self._assignment_switch_count

    def assignment_for(self, missing_id: Hashable) -> ReplacementAssignment | None:
        """返回指定缺失节点的分配；没有任务或未产生赢家时返回 None。"""
        return self._assignments.get(str(missing_id))

    def update_local_status(
        self, position_y: float, energy: float, neighbor_count: int, timestamp: float
    ) -> None:
        """更新本机竞价输入。

        参数依次是纵坐标、归一化能量、邻居数和本地时间；没有返回值。先推进时钟，
        使状态快照与心跳状态机处于相同时间点，避免用旧状态生成新竞价。
        """
        self._require_time(position_y, "position_y")
        self._require_time(energy, "energy")
        self._require_non_negative_int(neighbor_count, "neighbor_count")
        self.advance_time(timestamp)
        self._local_status = (float(position_y), float(energy), neighbor_count)

    def local_bid_for(
        self, missing_id: Hashable, current_load: int, timestamp: float
    ) -> ReplacementBid:
        """按配置公式创建并登记本机对一个缺失搜索带的竞价。

        参数为缺失节点、当前承担任务数和本地时间；返回含过期时刻的不可变竞价。
        公式使用固定搜索带，保证所有节点无需环境参与也能复现同一排序。
        """
        self._require_non_negative_int(current_load, "current_load")
        self.advance_time(timestamp)
        missing_key = str(missing_id)
        if self._local_status is None:
            raise ValueError("local status must be updated before bidding")
        if missing_key not in self._liveness:
            raise KeyError(missing_key)
        if missing_key == self.agent_id or self._liveness[missing_key] not in (
            NodeLiveness.MISSING,
            NodeLiveness.RECOVERING,
        ):
            raise ValueError("bid target must be a missing task other than bidder")
        if self._liveness[self.agent_id] is NodeLiveness.MISSING:
            raise ValueError("missing node cannot bid")

        position_y, energy, neighbor_count = self._local_status
        lane = self._lane_for(missing_key)
        energy_clipped = min(1.0, max(0.0, energy))
        score = (
            self.config.distance_weight * abs(position_y - lane.center_y) / self.world_height
            + self.config.load_weight * current_load
            + self.config.connectivity_weight * (1.0 / (1 + neighbor_count))
            + self.config.energy_weight * (1.0 - energy_clipped)
        )
        if not _is_finite_number(score):
            raise ValueError("bid score must be finite")
        bid = ReplacementBid(
            missing_id=missing_key,
            bidder_id=self.agent_id,
            score=float(score),
            created_at=self._last_time,
            expires_at=self._last_time + self.config.bid_ttl,
        )
        self._known_bids.setdefault(missing_key, {})[self.agent_id] = bid
        self._reconcile_assignments()
        return bid

    @property
    def missing_nodes(self) -> tuple[str, ...]:
        """返回当前已完成确认的缺失节点，顺序与固定成员表一致。"""
        return tuple(
            node_id
            for node_id in self.roster
            if self._liveness[node_id] is NodeLiveness.MISSING
        )

    def liveness_of(self, node_id: Hashable) -> NodeLiveness:
        """返回指定成员的存活状态；未知成员会引发 KeyError。"""
        return self._liveness[str(node_id)]

    def advance_time(self, timestamp: float) -> None:
        """将本地时钟单调推进到 timestamp，并更新超时状态机。"""
        self._require_time(timestamp, "timestamp")
        if timestamp < self._last_time:
            raise ValueError("timestamp must not move backwards")
        self._last_time = float(timestamp)
        # 本机每次推进时更新心跳，避免本机被远端缺失检测逻辑误判。
        self._last_heartbeats[self.agent_id] = self._last_time

        for node_id in self.roster:
            if node_id != self.agent_id:
                self._advance_liveness(node_id)
        self._reconcile_assignments()

    def build_gossip(self) -> dict[str, Any]:
        """构建心跳和每个任务当前最佳竞价，供邻居继续多跳转发。"""
        self._reconcile_assignments()
        return {
            "kind": "replacement_gossip",
            "sender": self.agent_id,
            "sent_at": self._last_time,
            "heartbeats": dict(self._last_heartbeats),
            "bids": {
                missing_id: self._bid_payload(min(bids.values(), key=lambda bid: (bid.score, str(bid.bidder_id))))
                for missing_id, bids in self._known_bids.items() if bids
            },
        }

    def ingest_gossip(self, message: object, received_at: float) -> bool:
        """校验并合并 gossip；合法消息返回 True，非法消息不改变本地状态。"""
        if not _is_finite_number(received_at) or not isinstance(message, Mapping):
            return False
        if set(message) != {"kind", "sender", "sent_at", "heartbeats", "bids"}:
            return False
        if message.get("kind") != "replacement_gossip":
            return False
        sender = message.get("sender")
        sent_at = message.get("sent_at")
        heartbeats = message.get("heartbeats")
        bids = message.get("bids")
        if (
            not isinstance(sender, str)
            or sender not in self._liveness
            or not _is_finite_number(sent_at)
            or sent_at > received_at
            or not isinstance(heartbeats, Mapping)
            or not isinstance(bids, Mapping)
        ):
            return False

        validated_heartbeats: dict[str, float] = {}
        for node_id, heartbeat_at in heartbeats.items():
            if (
                not isinstance(node_id, str)
                or node_id not in self._liveness
                or not _is_finite_number(heartbeat_at)
                or heartbeat_at > sent_at
                or heartbeat_at > received_at
            ):
                return False
            validated_heartbeats[node_id] = float(heartbeat_at)

        validated_bids: list[ReplacementBid] = []
        for missing_id, payload in bids.items():
            if (
                not isinstance(missing_id, str)
                or missing_id not in self._liveness
            ):
                return False
            # 分区两侧可能对同一成员持有不同存活视图。只要任务标识属于固定名单，
            # 就先验证整条竞价并合并心跳；本地仍健康的任务会在协调清理阶段丢弃。
            # 这样既不接受未知节点，也避免过时任务阻塞重连后的恢复证据。
            bid = self._parse_gossip_bid(
                missing_id, payload, float(sent_at), float(received_at), validated_heartbeats
            )
            if bid is None:
                return False
            validated_bids.append(bid)

        # 先验证完整消息再合并，避免非法数据留下任何部分状态变更。
        for node_id, heartbeat_at in validated_heartbeats.items():
            if heartbeat_at > self._last_heartbeats[node_id]:
                self._last_heartbeats[node_id] = heartbeat_at
                self._record_heartbeat(node_id, float(received_at))
        for bid in validated_bids:
            existing = self._known_bids.setdefault(str(bid.missing_id), {}).get(str(bid.bidder_id))
            if existing is None or bid.created_at > existing.created_at:
                self._known_bids[str(bid.missing_id)][str(bid.bidder_id)] = bid
        self._reconcile_assignments()
        return True

    def _reconcile_assignments(self) -> None:
        """清理无效任务和竞价，并在窗口到期后按稳定切换规则更新赢家。"""
        for missing_id in tuple(self._known_bids):
            if self._liveness[missing_id] is NodeLiveness.HEALTHY:
                self._known_bids.pop(missing_id, None)
                self._assignments.pop(missing_id, None)
                self._bid_window_started.pop(missing_id, None)
                self._last_winners.pop(missing_id, None)
                continue
            bids = self._known_bids[missing_id]
            for bidder_id, bid in tuple(bids.items()):
                if bid.expires_at <= self._last_time or self._liveness[bidder_id] is NodeLiveness.MISSING:
                    bids.pop(bidder_id)
            if not bids:
                self._known_bids.pop(missing_id, None)

        for missing_id in self.roster:
            if self._liveness[missing_id] is NodeLiveness.MISSING:
                self._bid_window_started.setdefault(missing_id, self._last_time)
            elif self._liveness[missing_id] is NodeLiveness.HEALTHY:
                self._assignments.pop(missing_id, None)
                self._known_bids.pop(missing_id, None)
                self._bid_window_started.pop(missing_id, None)
                self._last_winners.pop(missing_id, None)

        for missing_id, started_at in tuple(self._bid_window_started.items()):
            if self._liveness[missing_id] is NodeLiveness.HEALTHY or self._last_time - started_at < self.config.bid_window:
                continue
            candidates = self._known_bids.get(missing_id, {})
            winner = self._assignments.get(missing_id)
            winner_bid = candidates.get(str(winner.winner_id)) if winner else None
            if winner is not None and winner_bid is None:
                self._assignments.pop(missing_id, None)
                winner = None
            if not candidates:
                continue
            best = min(candidates.values(), key=lambda bid: (bid.score, str(bid.bidder_id)))
            if winner is None:
                self._assignments[missing_id] = ReplacementAssignment(
                    missing_id=missing_id, winner_id=best.bidder_id,
                    assigned_at=self._last_time, lane=self._lane_for(missing_id)
                )
                prior_winner = self._last_winners.get(missing_id)
                if prior_winner is not None and prior_winner != best.bidder_id:
                    self._assignment_switch_count += 1
                self._last_winners[missing_id] = str(best.bidder_id)
            elif best.bidder_id != winner.winner_id and best.score <= winner_bid.score - self.config.switch_margin:
                self._assignments[missing_id] = ReplacementAssignment(
                    missing_id=missing_id, winner_id=best.bidder_id,
                    assigned_at=self._last_time, lane=self._lane_for(missing_id)
                )
                self._assignment_switch_count += 1
                self._last_winners[missing_id] = str(best.bidder_id)

    def _parse_gossip_bid(
        self, missing_id: str, payload: object, sent_at: float, received_at: float,
        validated_heartbeats: Mapping[str, float],
    ) -> ReplacementBid | None:
        """验证一条线上的竞价负载并转换为对象；任何字段异常均拒绝整条 gossip。"""
        if not isinstance(payload, Mapping):
            return None
        required = ("bidder_id", "score", "created_at", "expires_at")
        bidder_id = payload.get("bidder_id")
        if set(payload) != set(required) or not isinstance(bidder_id, str) or bidder_id not in self._liveness or bidder_id == missing_id:
            return None
        # 同包新心跳会在原子提交时使竞价者进入恢复，故可作为其竞价仍可用的证据。
        if self._liveness[bidder_id] is NodeLiveness.MISSING and bidder_id not in validated_heartbeats:
            return None
        score, created_at, expires_at = (payload[name] for name in ("score", "created_at", "expires_at"))
        if (
            not _is_finite_number(score) or not _is_finite_number(created_at) or not _is_finite_number(expires_at)
            # 因果关系和 TTL 契约均必须采用发送方可复现的同一浮点表达式严格验证。
            or created_at > sent_at or created_at > received_at or expires_at <= received_at
            or expires_at != created_at + self.config.bid_ttl
        ):
            return None
        return ReplacementBid(missing_id, bidder_id, float(score), float(created_at), float(expires_at))

    @staticmethod
    def _bid_payload(bid: ReplacementBid) -> dict[str, Any]:
        """将最佳竞价转为固定外部协议，缺失节点由外层键唯一标识而不重复传输。"""
        return {"bidder_id": bid.bidder_id, "score": bid.score, "created_at": bid.created_at, "expires_at": bid.expires_at}

    def _lane_for(self, missing_id: str) -> CoverageLane:
        """按原始负责人标识返回其固定搜索带，成员表已在构造期验证完整。"""
        return next(lane for lane in self.lanes if lane.owner_id == missing_id)

    def _advance_liveness(self, node_id: str) -> None:
        """按当前时钟推进一个远端节点的超时、确认与恢复状态。"""
        state = self._liveness[node_id]
        heartbeat_age = self._last_time - self._last_heartbeats[node_id]
        if state is NodeLiveness.HEALTHY and heartbeat_age > self.config.failure_timeout:
            self._liveness[node_id] = NodeLiveness.SUSPECTED
            self._suspected_since[node_id] = self._last_time
        elif state is NodeLiveness.SUSPECTED:
            if heartbeat_age <= self.config.failure_timeout:
                # 新心跳在确认期内抵消怀疑，避免短暂网络抖动触发补位。
                self._liveness[node_id] = NodeLiveness.HEALTHY
                self._suspected_since.pop(node_id, None)
            elif (
                self._last_time - self._suspected_since[node_id]
                >= self.config.failure_confirmation
            ):
                self._liveness[node_id] = NodeLiveness.MISSING
                self._suspected_since.pop(node_id, None)
        elif state is NodeLiveness.RECOVERING:
            if heartbeat_age > self.config.failure_timeout:
                self._liveness[node_id] = NodeLiveness.SUSPECTED
                self._recovering_since.pop(node_id, None)
                self._suspected_since[node_id] = self._last_time
            elif (
                self._last_time - self._recovering_since[node_id]
                >= self.config.recovery_stability
            ):
                self._liveness[node_id] = NodeLiveness.HEALTHY
                self._recovering_since.pop(node_id, None)

    def _record_heartbeat(self, node_id: str, received_at: float) -> None:
        """将新心跳作用于状态机，使怀疑撤销或缺失节点进入恢复。"""
        state = self._liveness[node_id]
        if state is NodeLiveness.SUSPECTED:
            # 心跳已在消息校验阶段确认新鲜，接收时立即撤销短暂丢包造成的怀疑。
            self._liveness[node_id] = NodeLiveness.HEALTHY
            self._suspected_since.pop(node_id, None)
        elif state is NodeLiveness.MISSING:
            self._liveness[node_id] = NodeLiveness.RECOVERING
            # 稳定期从收到恢复证据起算，避免信任远端可伪造的网络时钟。
            self._recovering_since[node_id] = received_at

    @staticmethod
    def _require_non_negative_int(value: object, name: str) -> None:
        """验证计数为非布尔非负整数，避免 bool 被 Python 当作负载或邻居数。"""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _require_time(value: object, name: str) -> None:
        """验证时间参数为有限数值，避免状态机的比较失去全序。"""
        if not _is_finite_number(value):
            raise ValueError(f"{name} must be a finite number")


def build_coverage_lanes(roster: Sequence[Hashable], world_height: float) -> tuple[CoverageLane, ...]:
    """按稳定的代理标识顺序，为每个原始负责人划分等高搜索带。"""
    if not roster:
        raise ValueError("roster must not be empty")
    if (
        not _is_finite_number(world_height)
        or world_height <= 0
    ):
        raise ValueError("world_height must be a finite positive number")

    # 排序仅依赖字符串表示，保证各节点在同一名单上得到一致的固定搜索带。
    owners = sorted(str(agent_id) for agent_id in roster)
    lane_height = world_height / len(owners)
    return tuple(
        CoverageLane(
            lane_id=f"lane-{index}",
            owner_id=owner_id,
            center_y=(index + 0.5) * lane_height,
        )
        for index, owner_id in enumerate(owners)
    )
