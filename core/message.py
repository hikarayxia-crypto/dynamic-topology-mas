"""多智能体通信消息的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Hashable, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class Message:
    """智能体通过通信拓扑发送的消息。

    参数:
        sender: 发送方节点或智能体标识。
        receiver: 接收方；为 ``None`` 时表示广播给当前直接邻居。
        payload: 消息内容。
        created_at: 生成时间。
        ttl: 生存时间；为 ``None`` 时永不过期。
        message_id: 消息唯一标识，默认自动生成。
    """

    sender: Hashable
    receiver: Hashable | None
    payload: Any
    created_at: float
    ttl: float | None = None
    message_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sender is None:
            raise ValueError("sender 不能为 None")
        created_at = float(self.created_at)
        if not math.isfinite(created_at):
            raise ValueError("created_at 必须是有限数值")
        if self.ttl is not None:
            ttl = float(self.ttl)
            if not math.isfinite(ttl) or ttl < 0:
                raise ValueError("ttl 必须是非负有限数值或 None")
            object.__setattr__(self, "ttl", ttl)
        if not self.message_id:
            raise ValueError("message_id 不能为空")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def expires_at(self) -> float | None:
        """返回过期时刻；永不过期时返回 None。"""

        return None if self.ttl is None else self.created_at + self.ttl

    def is_expired(self, timestamp: float) -> bool:
        """判断消息在给定时刻是否已经失效。"""

        return self.expires_at is not None and float(timestamp) > self.expires_at
