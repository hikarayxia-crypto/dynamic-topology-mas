"""感知、通信和动作执行等智能体交互机制。"""

from .communication import (
    CommunicationBus,
    CommunicationConfig,
    CommunicationStats,
    TransmissionResult,
    TransmissionStatus,
)

__all__ = [
    "CommunicationBus",
    "CommunicationConfig",
    "CommunicationStats",
    "TransmissionResult",
    "TransmissionStatus",
]
