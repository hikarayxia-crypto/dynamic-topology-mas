"""动态拓扑与邻接关系管理。

该模块使用邻接表维护动态图，使用 NumPy 按需生成邻接矩阵。邻接表适合频繁
增删边，矩阵则方便后续 GNN、控制算法和实验指标直接使用。所有有效变更都会
记录时间戳和递增版本号，便于仿真环境查询某个时间段内发生的拓扑变化。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
from threading import RLock
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

NodeId = Hashable


class TopologyError(ValueError):
    """拓扑操作参数无效或破坏拓扑约束时抛出的异常。"""


class TopologyChangeType(str, Enum):
    """已经发生并写入历史记录的拓扑变化类型。"""

    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    NODE_ACTIVATED = "node_activated"
    NODE_DEACTIVATED = "node_deactivated"
    EDGE_CONNECTED = "edge_connected"
    EDGE_DISCONNECTED = "edge_disconnected"
    EDGE_RECONNECTED = "edge_reconnected"
    EDGE_UPDATED = "edge_updated"


class TopologyOperation(str, Enum):
    """可调度到未来仿真时刻执行的操作。"""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ACTIVATE_NODE = "activate_node"
    DEACTIVATE_NODE = "deactivate_node"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    RECONNECT = "reconnect"


@dataclass(frozen=True)
class Node:
    """拓扑节点。

    参数:
        node_id: 节点唯一标识，可使用整数或字符串。
        metadata: 节点能力、角色等扩展信息。
    """

    node_id: NodeId
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """通信边。

    参数:
        source: 起点；无向图中仅用于保存首次建边方向。
        target: 终点。
        weight: 正有限权重，可表示通信质量、距离或控制耦合强度。
        metadata: 链路带宽、时延等扩展信息。
    """

    source: NodeId
    target: NodeId
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyChange:
    """一次已提交的拓扑变化。"""

    version: int
    timestamp: float
    change_type: TopologyChangeType
    node_id: NodeId | None = None
    source: NodeId | None = None
    target: NodeId | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionStatus:
    """两个节点的连接状态查询结果。"""

    source_exists: bool
    target_exists: bool
    source_active: bool
    target_active: bool
    directly_connected: bool
    reachable: bool


@dataclass(order=True)
class _ScheduledOperation:
    """内部使用的定时操作；序号用于保证同一时刻按提交顺序执行。"""

    timestamp: float
    sequence: int
    operation: TopologyOperation = field(compare=False)
    arguments: dict[str, Any] = field(compare=False, default_factory=dict)


class DynamicTopology:
    """支持时间变化、状态查询和历史追踪的动态图。

    参数:
        directed: 是否使用有向通信图，默认使用无向图。
        initial_time: 仿真初始时间。
        allow_self_loops: 是否允许节点连接自身，默认禁止。

    关键逻辑:
        邻接表是唯一可变数据源；邻接矩阵由当前邻接表按需生成，从而避免每次
        增删节点或边都复制整张矩阵。节点暂时失效时保留原链路，恢复后可直接
        重新参与通信；节点永久移除时才删除所有关联边。
    """

    def __init__(
        self,
        directed: bool = False,
        initial_time: float = 0.0,
        allow_self_loops: bool = False,
    ) -> None:
        self.directed = directed
        self.allow_self_loops = allow_self_loops
        self._time = self._validate_time(initial_time)
        self._version = 0
        self._nodes: dict[NodeId, Node] = {}
        self._active_nodes: set[NodeId] = set()
        self._adjacency: dict[NodeId, dict[NodeId, Edge]] = {}
        self._disconnected_edges: dict[Any, Edge] = {}
        self._history: list[TopologyChange] = []
        self._schedule: list[_ScheduledOperation] = []
        self._schedule_sequence = 0
        self._lock = RLock()

    @property
    def current_time(self) -> float:
        """返回当前仿真时间。"""
        return self._time

    @property
    def version(self) -> int:
        """返回拓扑版本；每次有效变化后递增一次。"""
        return self._version

    @property
    def node_count(self) -> int:
        """返回所有未移除节点数，包括暂时失效节点。"""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """返回当前边数；无向边只计数一次。"""
        if self.directed:
            return sum(len(neighbors) for neighbors in self._adjacency.values())
        return len(
            {
                self._edge_key(source, target)
                for source, neighbors in self._adjacency.items()
                for target in neighbors
            }
        )

    def add_node(
        self,
        node_id: NodeId,
        metadata: Mapping[str, Any] | None = None,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """新增并激活节点；已存在时不产生变化并返回 False。"""
        self._validate_node_id(node_id)
        with self._lock:
            if node_id in self._nodes:
                return False
            event_time = self._resolve_event_time(timestamp)
            node = Node(node_id=node_id, metadata=dict(metadata or {}))
            self._nodes[node_id] = node
            self._active_nodes.add(node_id)
            self._adjacency[node_id] = {}
            self._record(
                TopologyChangeType.NODE_ADDED,
                event_time,
                node_id=node_id,
                details={"metadata": dict(node.metadata)},
            )
            return True

    def remove_node(
        self, node_id: NodeId, *, timestamp: float | None = None
    ) -> bool:
        """永久移除节点及其所有入边、出边；不存在时返回 False。"""
        with self._lock:
            if node_id not in self._nodes:
                return False
            event_time = self._resolve_event_time(timestamp)
            removed_edges = self._remove_incident_edges(node_id)
            del self._adjacency[node_id]
            del self._nodes[node_id]
            self._active_nodes.discard(node_id)
            self._discard_saved_edges_for_node(node_id)
            self._record(
                TopologyChangeType.NODE_REMOVED,
                event_time,
                node_id=node_id,
                details={"removed_edges": removed_edges},
            )
            return True

    def set_node_active(
        self,
        node_id: NodeId,
        active: bool,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """设置节点在线状态，用于模拟故障和恢复。

        暂时失效不删除原边，因此节点恢复后可以直接重新参与通信。
        """
        with self._lock:
            self._require_node(node_id)
            currently_active = node_id in self._active_nodes
            if currently_active == active:
                return False
            event_time = self._resolve_event_time(timestamp)
            if active:
                self._active_nodes.add(node_id)
                change_type = TopologyChangeType.NODE_ACTIVATED
            else:
                self._active_nodes.remove(node_id)
                change_type = TopologyChangeType.NODE_DEACTIVATED
            self._record(change_type, event_time, node_id=node_id)
            return True

    def connect(
        self,
        source: NodeId,
        target: NodeId,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """建立或更新通信边；完全相同的边已存在时返回 False。"""
        with self._lock:
            self._validate_edge_endpoints(source, target)
            edge = Edge(
                source=source,
                target=target,
                weight=self._validate_weight(weight),
                metadata=dict(metadata or {}),
            )
            existing = self._adjacency[source].get(target)
            if existing is not None:
                if existing.weight == edge.weight and dict(existing.metadata) == dict(edge.metadata):
                    return False
                event_time = self._resolve_event_time(timestamp)
                self._store_edge(edge)
                self._record(
                    TopologyChangeType.EDGE_UPDATED,
                    event_time,
                    source=source,
                    target=target,
                    details={"old_weight": existing.weight, "weight": edge.weight},
                )
                return True

            edge_key = self._edge_key(source, target)
            was_disconnected = edge_key in self._disconnected_edges
            event_time = self._resolve_event_time(timestamp)
            self._disconnected_edges.pop(edge_key, None)
            self._store_edge(edge)
            self._record(
                TopologyChangeType.EDGE_RECONNECTED
                if was_disconnected
                else TopologyChangeType.EDGE_CONNECTED,
                event_time,
                source=source,
                target=target,
                details={"weight": edge.weight, "metadata": dict(edge.metadata)},
            )
            return True

    def disconnect(
        self, source: NodeId, target: NodeId, *, timestamp: float | None = None
    ) -> bool:
        """断开一条边但保存其属性，以便后续重连。"""
        with self._lock:
            self._require_node(source)
            self._require_node(target)
            edge = self._adjacency[source].get(target)
            if edge is None:
                return False
            event_time = self._resolve_event_time(timestamp)
            del self._adjacency[source][target]
            if not self.directed:
                self._adjacency[target].pop(source, None)
            self._disconnected_edges[self._edge_key(source, target)] = edge
            self._record(
                TopologyChangeType.EDGE_DISCONNECTED,
                event_time,
                source=source,
                target=target,
                details={"weight": edge.weight},
            )
            return True

    def reconnect(
        self,
        source: NodeId,
        target: NodeId,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """按断开前的权重和元数据恢复链路。"""
        with self._lock:
            edge = self._disconnected_edges.get(self._edge_key(source, target))
            if edge is None:
                raise TopologyError(f"边 {source!r} -> {target!r} 没有可恢复的断开记录")
            return self.connect(
                source,
                target,
                weight=edge.weight,
                metadata=edge.metadata,
                timestamp=timestamp,
            )

    def has_node(self, node_id: NodeId) -> bool:
        """判断节点是否存在。"""
        return node_id in self._nodes

    def is_node_active(self, node_id: NodeId) -> bool:
        """判断节点是否存在且在线。"""
        return node_id in self._active_nodes

    def get_node(self, node_id: NodeId) -> Node:
        """返回节点；节点不存在时抛出 TopologyError。"""
        self._require_node(node_id)
        return self._nodes[node_id]

    def nodes(self, *, active_only: bool = False) -> tuple[NodeId, ...]:
        """按插入顺序返回节点标识。"""
        if not active_only:
            return tuple(self._nodes)
        return tuple(node_id for node_id in self._nodes if node_id in self._active_nodes)

    def get_neighbors(
        self,
        node_id: NodeId,
        *,
        active_only: bool = True,
        direction: str = "out",
    ) -> tuple[NodeId, ...]:
        """获取邻居集合。

        有向图的 ``direction`` 可选 ``out``、``in`` 或 ``all``。结果按节点
        插入顺序排列，避免集合遍历顺序导致实验不可复现。
        """
        self._require_node(node_id)
        if direction not in {"out", "in", "all"}:
            raise TopologyError("direction 必须是 'out'、'in' 或 'all'")
        if active_only and node_id not in self._active_nodes:
            return ()
        if not self.directed:
            # 无向边没有入/出方向，三种查询必须返回同一邻居集合。
            neighbor_set = set(self._adjacency[node_id])
        else:
            neighbor_set = (
                set(self._adjacency[node_id]) if direction in {"out", "all"} else set()
            )
        if self.directed and direction in {"in", "all"}:
            neighbor_set.update(
                source
                for source, neighbors in self._adjacency.items()
                if node_id in neighbors
            )
        if active_only:
            neighbor_set.intersection_update(self._active_nodes)
        return tuple(node for node in self._nodes if node in neighbor_set)

    def are_directly_connected(
        self, source: NodeId, target: NodeId, *, active_only: bool = True
    ) -> bool:
        """判断两个节点之间是否存在可用直连边。"""
        if source not in self._nodes or target not in self._nodes:
            return False
        if active_only and (
            source not in self._active_nodes or target not in self._active_nodes
        ):
            return False
        return target in self._adjacency[source]

    def is_reachable(
        self, source: NodeId, target: NodeId, *, active_only: bool = True
    ) -> bool:
        """使用广度优先搜索判断是否存在多跳通信路径。"""
        if source not in self._nodes or target not in self._nodes:
            return False
        if active_only and (
            source not in self._active_nodes or target not in self._active_nodes
        ):
            return False
        if source == target:
            return True
        visited = {source}
        queue: deque[NodeId] = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency[current]:
                if active_only and neighbor not in self._active_nodes:
                    continue
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def get_connection_status(
        self, source: NodeId, target: NodeId, *, active_only: bool = True
    ) -> ConnectionStatus:
        """同时返回节点状态、直连状态和多跳可达状态。"""
        return ConnectionStatus(
            source_exists=source in self._nodes,
            target_exists=target in self._nodes,
            source_active=source in self._active_nodes,
            target_active=target in self._active_nodes,
            directly_connected=self.are_directly_connected(
                source, target, active_only=active_only
            ),
            reachable=self.is_reachable(source, target, active_only=active_only),
        )

    def adjacency_matrix(
        self,
        node_order: Sequence[NodeId] | None = None,
        *,
        active_only: bool = True,
        dtype: np.dtype[Any] | type = np.float64,
    ) -> tuple[np.ndarray, tuple[NodeId, ...]]:
        """生成加权邻接矩阵并返回对应节点顺序。

        同时返回节点顺序是必要的：节点动态增删后，不能假设矩阵下标永远等于
        智能体编号。
        """
        if node_order is None:
            order = self.nodes(active_only=active_only)
        else:
            order = tuple(node_order)
            if len(set(order)) != len(order):
                raise TopologyError("node_order 不能包含重复节点")
            unknown = [node for node in order if node not in self._nodes]
            if unknown:
                raise TopologyError(f"node_order 包含未知节点: {unknown!r}")
            if active_only:
                order = tuple(node for node in order if node in self._active_nodes)
        index = {node: position for position, node in enumerate(order)}
        matrix = np.zeros((len(order), len(order)), dtype=dtype)
        for source in order:
            for target, edge in self._adjacency[source].items():
                target_index = index.get(target)
                if target_index is not None:
                    matrix[index[source], target_index] = edge.weight
        return matrix, order

    def get_changes(
        self,
        *,
        since_version: int = 0,
        since_time: float | None = None,
        change_types: Iterable[TopologyChangeType] | None = None,
    ) -> tuple[TopologyChange, ...]:
        """按版本、时间和类型增量查询拓扑变化。"""
        if since_version < 0:
            raise TopologyError("since_version 不能为负数")
        accepted_types = set(change_types) if change_types is not None else None
        return tuple(
            change
            for change in self._history
            if change.version > since_version
            and (since_time is None or change.timestamp >= since_time)
            and (accepted_types is None or change.change_type in accepted_types)
        )

    def schedule_change(
        self,
        timestamp: float,
        operation: TopologyOperation | str,
        **arguments: Any,
    ) -> None:
        """安排未来拓扑操作，参数通过 ``arguments`` 传给对应公开接口。"""
        event_time = self._validate_time(timestamp)
        if event_time < self._time:
            raise TopologyError("不能把拓扑变化安排到当前时间之前")
        try:
            parsed_operation = TopologyOperation(operation)
        except ValueError as exc:
            raise TopologyError(f"不支持的拓扑操作: {operation!r}") from exc
        with self._lock:
            self._schedule_sequence += 1
            heapq.heappush(
                self._schedule,
                _ScheduledOperation(
                    timestamp=event_time,
                    sequence=self._schedule_sequence,
                    operation=parsed_operation,
                    arguments=dict(arguments),
                ),
            )

    def advance_time(self, target_time: float) -> tuple[TopologyChange, ...]:
        """推进仿真时间、执行所有到期操作并返回本次产生的变化。"""
        target = self._validate_time(target_time)
        if target < self._time:
            raise TopologyError("仿真时间不能倒退")
        with self._lock:
            start_version = self._version
            while self._schedule and self._schedule[0].timestamp <= target:
                scheduled = heapq.heappop(self._schedule)
                self._execute_scheduled(scheduled)
            self._time = target
            return self.get_changes(since_version=start_version)

    def snapshot(self, *, active_only: bool = False) -> dict[str, Any]:
        """返回可序列化拓扑快照，供日志记录和实验复现使用。"""
        selected_nodes = self.nodes(active_only=active_only)
        selected_set = set(selected_nodes)
        edges: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for source in selected_nodes:
            for target, edge in self._adjacency[source].items():
                if target not in selected_set:
                    continue
                key = self._edge_key(source, target)
                if not self.directed and key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": edge.weight,
                        "metadata": dict(edge.metadata),
                    }
                )
        return {
            "time": self._time,
            "version": self._version,
            "directed": self.directed,
            "nodes": [
                {
                    "node_id": node_id,
                    "active": node_id in self._active_nodes,
                    "metadata": dict(self._nodes[node_id].metadata),
                }
                for node_id in selected_nodes
            ],
            "edges": edges,
        }

    def _execute_scheduled(self, scheduled: _ScheduledOperation) -> None:
        arguments = dict(scheduled.arguments)
        arguments["timestamp"] = scheduled.timestamp
        operation_map = {
            TopologyOperation.ADD_NODE: self.add_node,
            TopologyOperation.REMOVE_NODE: self.remove_node,
            TopologyOperation.CONNECT: self.connect,
            TopologyOperation.DISCONNECT: self.disconnect,
            TopologyOperation.RECONNECT: self.reconnect,
        }
        if scheduled.operation is TopologyOperation.ACTIVATE_NODE:
            self.set_node_active(arguments.pop("node_id"), True, **arguments)
        elif scheduled.operation is TopologyOperation.DEACTIVATE_NODE:
            self.set_node_active(arguments.pop("node_id"), False, **arguments)
        else:
            operation_map[scheduled.operation](**arguments)

    def _store_edge(self, edge: Edge) -> None:
        self._adjacency[edge.source][edge.target] = edge
        if not self.directed:
            self._adjacency[edge.target][edge.source] = edge

    def _remove_incident_edges(self, node_id: NodeId) -> int:
        edge_keys: set[Any] = set()
        for target in tuple(self._adjacency[node_id]):
            edge_keys.add(self._edge_key(node_id, target))
            if not self.directed:
                self._adjacency[target].pop(node_id, None)
        self._adjacency[node_id].clear()
        if self.directed:
            for source, neighbors in self._adjacency.items():
                if source != node_id and node_id in neighbors:
                    edge_keys.add(self._edge_key(source, node_id))
                    del neighbors[node_id]
        return len(edge_keys)

    def _discard_saved_edges_for_node(self, node_id: NodeId) -> None:
        self._disconnected_edges = {
            key: edge
            for key, edge in self._disconnected_edges.items()
            if edge.source != node_id and edge.target != node_id
        }

    def _record(
        self,
        change_type: TopologyChangeType,
        timestamp: float,
        *,
        node_id: NodeId | None = None,
        source: NodeId | None = None,
        target: NodeId | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._version += 1
        self._history.append(
            TopologyChange(
                version=self._version,
                timestamp=timestamp,
                change_type=change_type,
                node_id=node_id,
                source=source,
                target=target,
                details=dict(details or {}),
            )
        )

    def _resolve_event_time(self, timestamp: float | None) -> float:
        if timestamp is None:
            return self._time
        event_time = self._validate_time(timestamp)
        if event_time < self._time:
            raise TopologyError("拓扑变化时间不能早于当前仿真时间")
        self._time = event_time
        return event_time

    @staticmethod
    def _validate_time(timestamp: float) -> float:
        value = float(timestamp)
        if not math.isfinite(value):
            raise TopologyError("时间必须是有限数值")
        return value

    @staticmethod
    def _validate_weight(weight: float) -> float:
        value = float(weight)
        if not math.isfinite(value) or value <= 0:
            raise TopologyError("边权重必须是正有限数值")
        return value

    @staticmethod
    def _validate_node_id(node_id: NodeId) -> None:
        if node_id is None:
            raise TopologyError("node_id 不能为 None")
        try:
            hash(node_id)
        except TypeError as exc:
            raise TopologyError("node_id 必须是可哈希对象") from exc

    def _validate_edge_endpoints(self, source: NodeId, target: NodeId) -> None:
        self._require_node(source)
        self._require_node(target)
        if not self.allow_self_loops and source == target:
            raise TopologyError("当前拓扑禁止自环边")

    def _require_node(self, node_id: NodeId) -> None:
        if node_id not in self._nodes:
            raise TopologyError(f"节点不存在: {node_id!r}")

    def _edge_key(self, source: NodeId, target: NodeId) -> Any:
        return (source, target) if self.directed else frozenset((source, target))
