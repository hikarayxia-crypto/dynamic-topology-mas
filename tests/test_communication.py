"""动态拓扑通信总线测试。"""

import unittest

import numpy as np

from core.message import Message
from core.topology import DynamicTopology
from interaction.communication import (
    CommunicationBus,
    CommunicationConfig,
    TransmissionStatus,
)


class Receiver:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def receive_message(self, message: Message) -> None:
        self.messages.append(message)


class CommunicationBusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = DynamicTopology()
        for node_id in ("A", "B", "C"):
            self.topology.add_node(node_id)
        self.topology.connect("A", "B")
        self.topology.connect("A", "C")

    def test_delayed_message_is_delivered_at_due_time(self) -> None:
        bus = CommunicationBus(
            self.topology, CommunicationConfig(base_delay=1.0), seed=1
        )
        receiver = Receiver()
        result = bus.send(Message("A", "B", "hello", created_at=0.0))[0]
        self.assertEqual(result.status, TransmissionStatus.QUEUED)
        self.assertEqual(bus.advance_time(0.5, {"B": receiver}), ())
        delivered = bus.advance_time(1.0, {"B": receiver})
        self.assertEqual(delivered[0].status, TransmissionStatus.DELIVERED)
        self.assertEqual(receiver.messages[0].payload, "hello")
        self.assertEqual(bus.stats.attempted, 1)
        self.assertEqual(bus.stats.delivered, 1)

    def test_link_failure_drops_message_in_transit(self) -> None:
        bus = CommunicationBus(
            self.topology, CommunicationConfig(base_delay=1.0), seed=1
        )
        receiver = Receiver()
        bus.send(Message("A", "B", "data", created_at=0.0))
        self.topology.disconnect("A", "B")
        result = bus.advance_time(1.0, {"B": receiver})[0]
        self.assertEqual(result.status, TransmissionStatus.DROPPED)
        self.assertEqual(result.reason, "link_unavailable_on_delivery")
        self.assertEqual(receiver.messages, [])

    def test_broadcast_targets_current_direct_neighbors(self) -> None:
        bus = CommunicationBus(self.topology, seed=2)
        receivers = {"B": Receiver(), "C": Receiver()}
        queued = bus.send(Message("A", None, {"type": "status"}, created_at=0.0))
        self.assertEqual({result.receiver for result in queued}, {"B", "C"})
        delivered = bus.advance_time(0.0, receivers)
        self.assertEqual(len(delivered), 2)
        self.assertEqual(len(receivers["B"].messages), 1)
        self.assertEqual(len(receivers["C"].messages), 1)

    def test_disconnected_receiver_is_rejected(self) -> None:
        bus = CommunicationBus(self.topology)
        result = bus.send(Message("B", "C", "x", created_at=0.0))[0]
        self.assertEqual(result.status, TransmissionStatus.REJECTED)
        self.assertEqual(result.reason, "direct_link_unavailable")

    def test_random_packet_loss_and_expiration(self) -> None:
        loss_bus = CommunicationBus(
            self.topology, CommunicationConfig(packet_loss_rate=1.0), seed=3
        )
        loss = loss_bus.send(Message("A", "B", 1.0, created_at=0.0))[0]
        self.assertEqual(loss.status, TransmissionStatus.DROPPED)
        self.assertEqual(loss.reason, "random_packet_loss")

        expiry_bus = CommunicationBus(
            self.topology, CommunicationConfig(base_delay=1.0), seed=3
        )
        expiry_bus.send(Message("A", "B", 1.0, created_at=0.0, ttl=0.5))
        expired = expiry_bus.advance_time(1.0, {"B": Receiver()})[0]
        self.assertEqual(expired.reason, "expired")

    def test_numeric_noise_preserves_shape(self) -> None:
        bus = CommunicationBus(
            self.topology, CommunicationConfig(noise_std=0.2), seed=4
        )
        receiver = Receiver()
        payload = np.array([1.0, 2.0])
        bus.send(Message("A", "B", payload, created_at=0.0))
        bus.advance_time(0.0, {"B": receiver})
        noisy = receiver.messages[0].payload
        self.assertEqual(noisy.shape, payload.shape)
        self.assertFalse(np.array_equal(noisy, payload))

    def test_endpoint_can_use_different_topology_node_id(self) -> None:
        bus = CommunicationBus(self.topology)
        bus.register_endpoint("agent-a", "A")
        bus.register_endpoint("agent-b", "B")
        receiver = Receiver()
        result = bus.send(
            Message("agent-a", "agent-b", "mapped", created_at=0.0)
        )[0]
        self.assertEqual(result.status, TransmissionStatus.QUEUED)
        bus.advance_time(0.0, {"agent-b": receiver})
        self.assertEqual(receiver.messages[0].payload, "mapped")


if __name__ == "__main__":
    unittest.main()
