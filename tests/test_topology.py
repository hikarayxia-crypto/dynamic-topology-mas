"""动态拓扑模块的单元测试。"""

import unittest

import numpy as np

from core.topology import (
    DynamicTopology,
    TopologyChangeType,
    TopologyError,
    TopologyOperation,
)


class DynamicTopologyTests(unittest.TestCase):
    """验证节点、边、矩阵、查询、故障恢复和时间调度。"""

    def setUp(self) -> None:
        self.topology = DynamicTopology()
        for node_id in ("A", "B", "C"):
            self.topology.add_node(node_id)
        self.topology.connect("A", "B", weight=0.8)
        self.topology.connect("B", "C", weight=0.6)

    def test_neighbors_and_connection_status(self) -> None:
        self.assertEqual(self.topology.get_neighbors("B"), ("A", "C"))
        self.assertEqual(
            self.topology.get_neighbors("B", direction="in"), ("A", "C")
        )
        status = self.topology.get_connection_status("A", "C")
        self.assertFalse(status.directly_connected)
        self.assertTrue(status.reachable)

    def test_adjacency_matrix_keeps_node_mapping(self) -> None:
        matrix, order = self.topology.adjacency_matrix()
        self.assertEqual(order, ("A", "B", "C"))
        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [0.0, 0.8, 0.0],
                    [0.8, 0.0, 0.6],
                    [0.0, 0.6, 0.0],
                ]
            ),
        )

    def test_disconnect_and_reconnect_restore_edge(self) -> None:
        self.assertTrue(self.topology.disconnect("A", "B", timestamp=1.0))
        self.assertFalse(self.topology.are_directly_connected("A", "B"))
        self.assertFalse(self.topology.is_reachable("A", "C"))
        self.assertTrue(self.topology.reconnect("A", "B", timestamp=2.0))
        matrix, order = self.topology.adjacency_matrix()
        self.assertEqual(matrix[order.index("A"), order.index("B")], 0.8)
        self.assertEqual(
            self.topology.get_changes()[-1].change_type,
            TopologyChangeType.EDGE_RECONNECTED,
        )

    def test_node_failure_and_recovery_preserve_edges(self) -> None:
        self.topology.set_node_active("B", False, timestamp=1.0)
        self.assertEqual(self.topology.get_neighbors("A"), ())
        self.assertEqual(self.topology.get_neighbors("B"), ())
        self.assertFalse(self.topology.is_reachable("A", "C"))
        active_matrix, active_order = self.topology.adjacency_matrix()
        self.assertEqual(active_order, ("A", "C"))
        self.assertEqual(active_matrix.shape, (2, 2))
        self.topology.set_node_active("B", True, timestamp=2.0)
        self.assertTrue(self.topology.is_reachable("A", "C"))
        self.assertEqual(self.topology.get_neighbors("B"), ("A", "C"))

    def test_remove_node_clears_incident_edges(self) -> None:
        self.assertTrue(self.topology.remove_node("B", timestamp=1.0))
        self.assertEqual(self.topology.nodes(), ("A", "C"))
        self.assertEqual(self.topology.edge_count, 0)
        self.assertEqual(self.topology.get_neighbors("A"), ())
        self.assertEqual(
            self.topology.get_changes()[-1].details["removed_edges"], 2
        )

    def test_scheduled_changes_execute_in_time_order(self) -> None:
        self.topology.schedule_change(
            2.0,
            TopologyOperation.ADD_NODE,
            node_id="D",
            metadata={"role": "relay"},
        )
        self.topology.schedule_change(
            3.0, TopologyOperation.CONNECT, source="C", target="D", weight=0.9
        )
        self.topology.schedule_change(
            4.0, TopologyOperation.DISCONNECT, source="B", target="C"
        )
        first_changes = self.topology.advance_time(2.5)
        self.assertEqual(len(first_changes), 1)
        self.assertTrue(self.topology.has_node("D"))
        self.assertFalse(self.topology.are_directly_connected("C", "D"))
        later_changes = self.topology.advance_time(4.0)
        self.assertEqual(len(later_changes), 2)
        self.assertTrue(self.topology.are_directly_connected("C", "D"))
        self.assertFalse(self.topology.are_directly_connected("B", "C"))
        self.assertEqual(self.topology.current_time, 4.0)

    def test_incremental_change_query(self) -> None:
        consumed_version = self.topology.version
        self.topology.disconnect("A", "B", timestamp=1.0)
        self.topology.add_node("D", timestamp=1.5)
        changes = self.topology.get_changes(since_version=consumed_version)
        self.assertEqual(
            [change.change_type for change in changes],
            [
                TopologyChangeType.EDGE_DISCONNECTED,
                TopologyChangeType.NODE_ADDED,
            ],
        )

    def test_invalid_operations_raise_clear_errors(self) -> None:
        original_time = self.topology.current_time
        with self.assertRaises(TopologyError):
            self.topology.connect("A", "missing", timestamp=10.0)
        with self.assertRaises(TopologyError):
            self.topology.connect("A", "A")
        with self.assertRaises(TopologyError):
            self.topology.reconnect("A", "C")
        with self.assertRaises(TopologyError):
            self.topology.advance_time(-1.0)
        # 失败操作不应偷偷推进仿真时钟，否则后续合法事件可能无法调度。
        self.assertEqual(self.topology.current_time, original_time)

    def test_directed_topology_distinguishes_in_and_out_neighbors(self) -> None:
        directed = DynamicTopology(directed=True)
        for node_id in (1, 2, 3):
            directed.add_node(node_id)
        directed.connect(1, 2)
        directed.connect(3, 2)
        self.assertEqual(directed.get_neighbors(2, direction="in"), (1, 3))
        self.assertEqual(directed.get_neighbors(2, direction="out"), ())
        self.assertTrue(directed.is_reachable(1, 2))
        self.assertFalse(directed.is_reachable(2, 1))


if __name__ == "__main__":
    unittest.main()
