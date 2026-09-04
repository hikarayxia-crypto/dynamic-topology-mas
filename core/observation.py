"""智能体自身、邻域和任务观测的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


def _feature_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """转换并校验一维特征，集中处理维度和数值有效性。"""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} 必须是一维向量")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 不能包含 NaN 或无穷值")
    return vector.copy()


@dataclass(frozen=True)
class NeighborObservation:
    """一个邻居在当前通信拓扑下可被观测到的特征。"""

    neighbor_id: Hashable
    features: np.ndarray
    link_weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.neighbor_id is None:
            raise ValueError("neighbor_id 不能为 None")
        object.__setattr__(
            self, "features", _feature_vector(self.features, "邻居特征")
        )
        weight = float(self.link_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("link_weight 必须是正有限数值")
        object.__setattr__(self, "link_weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class Observation:
    """单个智能体在一个时间步获得的结构化观测。

    参数:
        agent_id: 观测所属智能体。
        self_features: 自身状态向量。
        neighbors: 数量可变的邻域观测。
        task_features: 目标、覆盖状态等任务相关向量。
        topology_version: 生成观测时使用的拓扑版本。
        timestamp: 观测对应的仿真时间。
        neighbor_feature_dim: 无邻居时仍用于确定空矩阵的列数。

    设计说明:
        邻居保持为可变长度集合，而不是填充到固定智能体数量，从数据层面避免
        策略依赖特定规模；训练阶段可在批处理边界统一添加掩码。
    """

    agent_id: Hashable
    self_features: np.ndarray
    neighbors: tuple[NeighborObservation, ...] = ()
    task_features: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    topology_version: int = 0
    timestamp: float = 0.0
    neighbor_feature_dim: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.agent_id is None:
            raise ValueError("agent_id 不能为 None")
        self_features = _feature_vector(self.self_features, "自身特征")
        task_features = _feature_vector(self.task_features, "任务特征")
        neighbors = tuple(self.neighbors)
        if self.topology_version < 0:
            raise ValueError("topology_version 不能为负数")
        if self.neighbor_feature_dim < 0:
            raise ValueError("neighbor_feature_dim 不能为负数")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("观测时间必须是有限数值")

        inferred_dim = neighbors[0].features.size if neighbors else self.neighbor_feature_dim
        if neighbors and self.neighbor_feature_dim not in {0, inferred_dim}:
            raise ValueError("neighbor_feature_dim 与实际邻居特征维度不一致")
        if any(neighbor.features.size != inferred_dim for neighbor in neighbors):
            raise ValueError("同一观测中的邻居特征维度必须一致")
        neighbor_ids = [neighbor.neighbor_id for neighbor in neighbors]
        if len(set(neighbor_ids)) != len(neighbor_ids):
            raise ValueError("neighbors 不能包含重复邻居")

        object.__setattr__(self, "self_features", self_features)
        object.__setattr__(self, "task_features", task_features)
        object.__setattr__(self, "neighbors", neighbors)
        object.__setattr__(self, "neighbor_feature_dim", inferred_dim)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def neighbor_matrix(self, *, weighted: bool = False) -> np.ndarray:
        """返回 ``邻居数 × 特征维数`` 矩阵。

        ``weighted=True`` 时乘以链路权重，便于后续图聚合使用通信质量信息。
        """

        if not self.neighbors:
            return np.empty((0, self.neighbor_feature_dim), dtype=np.float64)
        matrix = np.stack([neighbor.features for neighbor in self.neighbors])
        if weighted:
            weights = np.asarray(
                [neighbor.link_weight for neighbor in self.neighbors], dtype=np.float64
            )
            matrix = matrix * weights[:, None]
        return matrix

    def mean_pooled_vector(self, *, weighted: bool = False) -> np.ndarray:
        """拼接自身、邻域均值和任务特征，提供无神经网络依赖的基线表示。"""

        matrix = self.neighbor_matrix(weighted=weighted)
        neighbor_mean = (
            matrix.mean(axis=0)
            if matrix.shape[0] > 0
            else np.zeros(self.neighbor_feature_dim, dtype=np.float64)
        )
        return np.concatenate((self.self_features, neighbor_mean, self.task_features))
