"""动态拓扑最小演示，可用于人工检查节点和链路随时间变化。"""

from pathlib import Path
import sys

# 兼容 VS Code 的“运行 Python 文件”操作：脚本目录启动时显式加入项目根目录。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.topology import DynamicTopology, TopologyOperation


def main() -> None:
    """构造三节点网络，并依次模拟断链、新节点加入与重连。"""
    topology = DynamicTopology()
    for node_id in ("UAV-1", "UAV-2", "UAV-3"):
        topology.add_node(node_id)
    topology.connect("UAV-1", "UAV-2")
    topology.connect("UAV-2", "UAV-3")

    topology.schedule_change(
        1.0,
        TopologyOperation.DISCONNECT,
        source="UAV-2",
        target="UAV-3",
    )
    topology.schedule_change(2.0, TopologyOperation.ADD_NODE, node_id="UAV-4")
    topology.schedule_change(
        2.0,
        TopologyOperation.CONNECT,
        source="UAV-1",
        target="UAV-4",
        weight=0.9,
    )
    topology.schedule_change(
        3.0,
        TopologyOperation.RECONNECT,
        source="UAV-2",
        target="UAV-3",
    )

    for target_time in (0.0, 1.0, 2.0, 3.0):
        changes = topology.advance_time(target_time)
        matrix, order = topology.adjacency_matrix()
        print(f"\n时间 {target_time:.1f}，节点顺序: {order}")
        print(matrix)
        print("本时刻变化:", [change.change_type.value for change in changes])


if __name__ == "__main__":
    main()
