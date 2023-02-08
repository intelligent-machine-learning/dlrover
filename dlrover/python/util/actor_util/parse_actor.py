from dlrover.python.common.constants import NodeType

def parse_type(name) -> str:
    name = name.lower()
    node_type: str = ""
    if NodeType.PS in name:
        node_type = NodeType.PS
    elif NodeType.EVALUATOR in name:
        node_type = NodeType.EVALUATOR
    elif NodeType.WORKER in name:
        node_type = NodeType.WORKER
    return node_type


def parse_index(name) -> int:
    """
    PsActor_1 split("_")[-1]
    TFSinkFunction-4|20 split("|").split("-")[-1]
    """
    node_type = parse_type(name)
    node_index: int = 0
    if node_type == NodeType.PS:
        node_index = int(name.split("_")[-1])
    elif node_type == NodeType.EVALUATOR:
        node_index = 1
    elif node_type == NodeType.WORKER:
        node_index = int(name.split("|")[0].split("-")[-1])
    return node_index


def parse_type_id_from_actor_name(name):
    node_type = parse_type(name)
    node_index = parse_index(name)
    return node_type, node_index