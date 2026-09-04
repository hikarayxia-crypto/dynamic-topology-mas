"""将数量可变的邻域观测转换为固定维度图表示。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from core.observation import Observation


@dataclass(frozen=True)
class GraphBatch:
    """经过填充的图观测批次。

    ``neighbor_mask`` 标出真实邻居，确保填充值不会参与聚合；链路权重只在真实
    边上生效，因此断链或节点离线会立即反映到编码结果中。
    """

    self_features: Tensor
    neighbor_features: Tensor
    neighbor_weights: Tensor
    neighbor_mask: Tensor
    task_features: Tensor

    def to(self, device: torch.device | str) -> "GraphBatch":
        """返回移动到目标设备的新批次。"""

        return GraphBatch(*(value.to(device) for value in self.__dict__.values()))


def batch_observations(
    observations: Sequence[Observation],
    *,
    device: torch.device | str = "cpu",
) -> GraphBatch:
    """把可变邻居观测填充成张量批次。

    参数:
        observations: 同一任务定义下的一组智能体观测。
        device: 张量所在设备。

    返回值:
        可直接输入图编码器的 ``GraphBatch``。
    """

    if not observations:
        raise ValueError("observations 不能为空")
    self_dim = observations[0].self_features.size
    neighbor_dim = observations[0].neighbor_feature_dim
    task_dim = observations[0].task_features.size
    if any(obs.self_features.size != self_dim for obs in observations):
        raise ValueError("批次中的自身特征维度必须一致")
    if any(obs.neighbor_feature_dim != neighbor_dim for obs in observations):
        raise ValueError("批次中的邻居特征维度必须一致")
    if any(obs.task_features.size != task_dim for obs in observations):
        raise ValueError("批次中的任务特征维度必须一致")

    batch_size = len(observations)
    # 至少保留一个填充槽，统一处理所有智能体都暂时失联的情况。
    max_neighbors = max(1, max(len(obs.neighbors) for obs in observations))
    self_features = torch.zeros((batch_size, self_dim), dtype=torch.float32)
    neighbor_features = torch.zeros(
        (batch_size, max_neighbors, neighbor_dim), dtype=torch.float32
    )
    neighbor_weights = torch.zeros((batch_size, max_neighbors), dtype=torch.float32)
    neighbor_mask = torch.zeros((batch_size, max_neighbors), dtype=torch.bool)
    task_features = torch.zeros((batch_size, task_dim), dtype=torch.float32)

    for row, observation in enumerate(observations):
        self_features[row] = torch.as_tensor(observation.self_features)
        task_features[row] = torch.as_tensor(observation.task_features)
        for column, neighbor in enumerate(observation.neighbors):
            neighbor_features[row, column] = torch.as_tensor(neighbor.features)
            neighbor_weights[row, column] = float(neighbor.link_weight)
            neighbor_mask[row, column] = True

    return GraphBatch(
        self_features.to(device),
        neighbor_features.to(device),
        neighbor_weights.to(device),
        neighbor_mask.to(device),
        task_features.to(device),
    )


class GraphObservationEncoder(nn.Module):
    """使用共享消息网络和加权均值完成拓扑无关编码。

    所有边复用同一个邻居编码器，聚合采用对排列不敏感的加权均值。因此模型
    参数量不随邻居数或智能体总数变化，可直接用于断链、节点增减和规模泛化。
    """

    def __init__(
        self,
        self_dim: int,
        neighbor_dim: int,
        task_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(self_dim, neighbor_dim, task_dim, hidden_dim) <= 0:
            raise ValueError("编码器维度必须为正整数")
        self.self_dim = self_dim
        self.neighbor_dim = neighbor_dim
        self.task_dim = task_dim
        self.hidden_dim = hidden_dim
        self.self_encoder = nn.Sequential(nn.Linear(self_dim, hidden_dim), nn.Tanh())
        self.message_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_dim), nn.Tanh()
        )
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden_dim), nn.Tanh())
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh()
        )

    def forward(self, batch: GraphBatch) -> Tensor:
        """返回形状为 ``[批量大小, hidden_dim]`` 的局部图表示。"""

        if batch.self_features.shape[-1] != self.self_dim:
            raise ValueError("自身特征维度与编码器配置不一致")
        if batch.neighbor_features.shape[-1] != self.neighbor_dim:
            raise ValueError("邻居特征维度与编码器配置不一致")
        if batch.task_features.shape[-1] != self.task_dim:
            raise ValueError("任务特征维度与编码器配置不一致")

        self_embedding = self.self_encoder(batch.self_features)
        messages = self.message_encoder(batch.neighbor_features)
        effective_weights = (
            batch.neighbor_weights * batch.neighbor_mask.to(batch.neighbor_weights.dtype)
        )
        weighted_sum = (messages * effective_weights.unsqueeze(-1)).sum(dim=1)
        weight_sum = effective_weights.sum(dim=1, keepdim=True)
        # 无邻居时显式输出零消息，避免除零且不会虚构邻居信息。
        neighbor_embedding = weighted_sum / weight_sum.clamp_min(1e-8)
        neighbor_embedding = torch.where(
            weight_sum > 0, neighbor_embedding, torch.zeros_like(neighbor_embedding)
        )
        task_embedding = self.task_encoder(batch.task_features)
        return self.output(
            torch.cat((self_embedding, neighbor_embedding, task_embedding), dim=-1)
        )
