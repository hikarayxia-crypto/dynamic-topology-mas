"""基于动态拓扑的消息路由、丢包、时延与噪声模拟。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import heapq
import math
from typing import Any, Hashable, Mapping, Protocol

import numpy as np

from core.message import Message
from core.topology import DynamicTopology, TopologyError


class MessageReceiver(Protocol):
    """通信总线可投递消息的最小接收接口。"""

    def receive_message(self, message: Message) -> None: ...


class TransmissionStatus(str, Enum):
    """一次单播传输当前所处的状态。"""

    QUEUED = "queued"
    DELIVERED = "delivered"
    DROPPED = "dropped"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TransmissionResult:
    """一次传输尝试的结果，用于日志和丢包率指标。"""

    message_id: str
    sender: Hashable
    receiver: Hashable | None
    status: TransmissionStatus
    timestamp: float
    reason: str | None = None
    deliver_at: float | None = None


@dataclass(frozen=True)
class CommunicationConfig:
    """通信扰动参数。

    参数:
        base_delay: 固定传输时延。
        delay_jitter: 均匀时延抖动范围，实际时延不会小于零。
        packet_loss_rate: 独立随机丢包率。
        noise_std: 数值载荷的零均值高斯噪声标准差。
        revalidate_on_delivery: 投递前是否再次检查链路；启用后可反映传输途中断链。
    """

    base_delay: float = 0.0
    delay_jitter: float = 0.0
    packet_loss_rate: float = 0.0
    noise_std: float = 0.0
    revalidate_on_delivery: bool = True

    def __post_init__(self) -> None:
        for name in ("base_delay", "delay_jitter", "noise_std"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数值")
            object.__setattr__(self, name, value)
        loss = float(self.packet_loss_rate)
        if not math.isfinite(loss) or not 0.0 <= loss <= 1.0:
            raise ValueError("packet_loss_rate 必须位于 [0, 1]")
        object.__setattr__(self, "packet_loss_rate", loss)


@dataclass
class CommunicationStats:
    """通信总线累计统计量。"""

    attempted: int = 0
    queued: int = 0
    delivered: int = 0
    dropped: int = 0
    rejected: int = 0

    @property
    def packet_loss_ratio(self) -> float:
        """返回已确定结果中的丢包比例，不把无效请求计入分母。"""

        completed = self.delivered + self.dropped
        return self.dropped / completed if completed else 0.0


@dataclass(order=True)
class _QueuedMessage:
    """内部优先队列项，同一时刻按发送顺序稳定投递。"""

    deliver_at: float
    sequence: int
    message: Message = field(compare=False)
    sender_node: Hashable = field(compare=False)
    receiver_node: Hashable = field(compare=False)


class CommunicationBus:
    """根据 ``DynamicTopology`` 路由直接邻居消息。

    端点通常是智能体 ID，拓扑节点可以使用不同 ID；通过 ``register_endpoint``
    建立一一映射。若未注册且拓扑中存在同名节点，则自动采用同名映射。
    """

    def __init__(
        self,
        topology: DynamicTopology,
        config: CommunicationConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.topology = topology
        self.config = config or CommunicationConfig()
        self.stats = CommunicationStats()
        self._rng = np.random.default_rng(seed)
        self._queue: list[_QueuedMessage] = []
        self._sequence = 0
        self._time = topology.current_time
        self._endpoint_nodes: dict[Hashable, Hashable] = {}
        self._node_endpoints: dict[Hashable, Hashable] = {}

    @property
    def current_time(self) -> float:
        """返回通信总线已推进到的时间。"""

        return self._time

    @property
    def pending_count(self) -> int:
        """返回等待投递的消息数。"""

        return len(self._queue)

    def register_endpoint(
        self, endpoint_id: Hashable, node_id: Hashable | None = None
    ) -> None:
        """把通信端点绑定到一个拓扑节点。

        一节点只允许一个端点，避免广播时无法确定消息应该投递给哪个智能体。
        """

        resolved_node = endpoint_id if node_id is None else node_id
        if not self.topology.has_node(resolved_node):
            raise TopologyError(f"不能绑定不存在的拓扑节点: {resolved_node!r}")
        existing_endpoint = self._node_endpoints.get(resolved_node)
        if existing_endpoint is not None and existing_endpoint != endpoint_id:
            raise ValueError(f"拓扑节点已绑定端点: {resolved_node!r}")
        old_node = self._endpoint_nodes.get(endpoint_id)
        if old_node is not None and old_node != resolved_node:
            self._node_endpoints.pop(old_node, None)
        self._endpoint_nodes[endpoint_id] = resolved_node
        self._node_endpoints[resolved_node] = endpoint_id

    def unregister_endpoint(self, endpoint_id: Hashable) -> bool:
        """解除端点映射；不存在时返回 False。"""

        node_id = self._endpoint_nodes.pop(endpoint_id, None)
        if node_id is None:
            return False
        self._node_endpoints.pop(node_id, None)
        return True

    def send(
        self, message: Message, *, current_time: float | None = None
    ) -> tuple[TransmissionResult, ...]:
        """发送单播或直接邻居广播消息。

        返回值:
            每个目标一个结果。成功接收前状态为 ``QUEUED``；无链路、节点离线、
            随机丢包等情况会立即给出 ``REJECTED`` 或 ``DROPPED``。
        """

        now = self._validated_time(self._time if current_time is None else current_time)
        if now < self._time:
            raise ValueError("通信时间不能倒退")
        if message.created_at > now:
            raise ValueError("不能在消息创建时间之前发送")
        self._time = now

        sender_node = self._resolve_node(message.sender)
        if sender_node is None or not self.topology.is_node_active(sender_node):
            self.stats.attempted += 1
            return (
                self._result(
                    message,
                    message.receiver,
                    TransmissionStatus.REJECTED,
                    now,
                    "sender_unavailable",
                ),
            )

        if message.receiver is None:
            target_endpoints = self._broadcast_targets(sender_node)
            if not target_endpoints:
                self.stats.attempted += 1
                return (
                    self._result(
                        message,
                        None,
                        TransmissionStatus.REJECTED,
                        now,
                        "no_available_neighbors",
                    ),
                )
        else:
            target_endpoints = (message.receiver,)

        results = [
            self._send_to_target(message, target, sender_node, now)
            for target in target_endpoints
        ]
        return tuple(results)

    def advance_time(
        self,
        timestamp: float,
        receivers: Mapping[Hashable, MessageReceiver],
    ) -> tuple[TransmissionResult, ...]:
        """推进通信时间并投递所有到期消息。

        投递时重新检查节点和边，是为了让发送后发生的故障、断链真正影响尚在
        传输中的消息，而不是只在发送瞬间检查一次。
        """

        now = self._validated_time(timestamp)
        if now < self._time:
            raise ValueError("通信时间不能倒退")
        self._time = now
        results: list[TransmissionResult] = []
        while self._queue and self._queue[0].deliver_at <= now:
            queued = heapq.heappop(self._queue)
            message = queued.message
            reason: str | None = None
            receiver = receivers.get(message.receiver)
            if message.is_expired(now):
                reason = "expired"
            elif receiver is None:
                reason = "receiver_missing"
            elif self.config.revalidate_on_delivery and not self._link_available(
                queued.sender_node, queued.receiver_node
            ):
                reason = "link_unavailable_on_delivery"

            if reason is not None:
                self.stats.dropped += 1
                results.append(
                    self._result(
                        message,
                        message.receiver,
                        TransmissionStatus.DROPPED,
                        now,
                        reason,
                        queued.deliver_at,
                    )
                )
                continue

            receiver.receive_message(message)
            self.stats.delivered += 1
            results.append(
                self._result(
                    message,
                    message.receiver,
                    TransmissionStatus.DELIVERED,
                    now,
                    deliver_at=queued.deliver_at,
                )
            )
        return tuple(results)

    def _send_to_target(
        self,
        original: Message,
        target_endpoint: Hashable,
        sender_node: Hashable,
        now: float,
    ) -> TransmissionResult:
        self.stats.attempted += 1
        receiver_node = self._resolve_node(target_endpoint)
        if receiver_node is None:
            return self._result(
                original,
                target_endpoint,
                TransmissionStatus.REJECTED,
                now,
                "receiver_unknown",
            )
        if not self._link_available(sender_node, receiver_node):
            return self._result(
                original,
                target_endpoint,
                TransmissionStatus.REJECTED,
                now,
                "direct_link_unavailable",
            )
        if original.is_expired(now):
            self.stats.dropped += 1
            return self._result(
                original,
                target_endpoint,
                TransmissionStatus.DROPPED,
                now,
                "expired",
            )
        if self._rng.random() < self.config.packet_loss_rate:
            self.stats.dropped += 1
            return self._result(
                original,
                target_endpoint,
                TransmissionStatus.DROPPED,
                now,
                "random_packet_loss",
            )

        jitter = (
            self._rng.uniform(-self.config.delay_jitter, self.config.delay_jitter)
            if self.config.delay_jitter
            else 0.0
        )
        deliver_at = now + max(0.0, self.config.base_delay + jitter)
        message = replace(
            original,
            receiver=target_endpoint,
            payload=self._apply_noise(original.payload),
        )
        self._sequence += 1
        heapq.heappush(
            self._queue,
            _QueuedMessage(
                deliver_at=deliver_at,
                sequence=self._sequence,
                message=message,
                sender_node=sender_node,
                receiver_node=receiver_node,
            ),
        )
        self.stats.queued += 1
        return self._result(
            message,
            target_endpoint,
            TransmissionStatus.QUEUED,
            now,
            deliver_at=deliver_at,
        )

    def _broadcast_targets(self, sender_node: Hashable) -> tuple[Hashable, ...]:
        targets: list[Hashable] = []
        for neighbor_node in self.topology.get_neighbors(sender_node):
            endpoint = self._node_endpoints.get(neighbor_node)
            if endpoint is None and self.topology.has_node(neighbor_node):
                endpoint = neighbor_node
            targets.append(endpoint)
        return tuple(targets)

    def _resolve_node(self, endpoint_id: Hashable) -> Hashable | None:
        node_id = self._endpoint_nodes.get(endpoint_id, endpoint_id)
        return node_id if self.topology.has_node(node_id) else None

    def _link_available(self, sender_node: Hashable, receiver_node: Hashable) -> bool:
        return self.topology.are_directly_connected(
            sender_node, receiver_node, active_only=True
        )

    def _apply_noise(self, payload: Any) -> Any:
        if self.config.noise_std == 0:
            return payload
        if isinstance(payload, np.ndarray) and np.issubdtype(payload.dtype, np.number):
            noise = self._rng.normal(0.0, self.config.noise_std, payload.shape)
            return payload.astype(np.float64, copy=True) + noise
        if isinstance(payload, (int, float, np.number)) and not isinstance(payload, bool):
            return float(payload) + float(self._rng.normal(0.0, self.config.noise_std))
        # 非数值载荷无法定义高斯噪声，保持原值并让上层自行编码。
        return payload

    def _result(
        self,
        message: Message,
        receiver: Hashable | None,
        status: TransmissionStatus,
        timestamp: float,
        reason: str | None = None,
        deliver_at: float | None = None,
    ) -> TransmissionResult:
        if status is TransmissionStatus.REJECTED:
            self.stats.rejected += 1
        return TransmissionResult(
            message_id=message.message_id,
            sender=message.sender,
            receiver=receiver,
            status=status,
            timestamp=timestamp,
            reason=reason,
            deliver_at=deliver_at,
        )

    @staticmethod
    def _validated_time(timestamp: float) -> float:
        value = float(timestamp)
        if not math.isfinite(value):
            raise ValueError("通信时间必须是有限数值")
        return value
