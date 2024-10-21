import os
from typing import List, Tuple
from ppievo.io import Tree

TransitionsType = List[Tuple[str, str, float]]


def read_transitions(
    transitions_path: str,
) -> TransitionsType:
    transitions = []
    lines = open(transitions_path, "r").read().strip().split("\n")
    if len(lines) == 0:
        raise Exception(f"The transitions file at {transitions_path} is empty")
    for i, line in enumerate(lines):
        if i == 0:
            tokens = line.split(" ")
            if len(tokens) != 2:
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if tokens[1] != "transitions":
                raise ValueError(
                    f"Transitions file at '{transitions_path}' should start "
                    f"with '[NUM_TRANSITIONS] transitions'."
                )
            if len(lines) - 1 != int(tokens[0]):
                raise ValueError(
                    f"Expected {int(tokens[0])} transitions at "
                    f"'{transitions_path}', but found only {len(lines) - 1}."
                )
        else:
            x1, y1, x2, y2, tx_str, ty_str = line.split(" ")
            tx, ty = float(tx_str), float(ty_str)
            transitions.append((x1, y1, x2, y2, tx, ty))
    return transitions


def write_transitions(transitions: TransitionsType, transitions_path: str) -> None:
    transitions_dir = os.path.dirname(transitions_path)
    if not os.path.exists(transitions_dir):
        os.makedirs(transitions_dir)
    res = (
        f"{len(transitions)} transitions\n"
        + "\n".join(
            [
                f"{x1} {y1} {x2} {y2} {tx} {ty}"
                for (x1, y1, x2, y2, tx, ty) in transitions
            ]
        )
        + "\n"
    )
    with open(transitions_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def get_leaf_distance(tree: Tree, leaf1: str, leaf2: str) -> float:
    """
    Calculate the distance between two leaf nodes in a tree
    """
    if not tree.is_leaf(leaf1) or not tree.is_leaf(leaf2):
        raise ValueError("Both nodes must be leaves")

    # Helper function to get path from root to a node
    def path_to_root(node):
        path = []
        while not tree.is_root(node):
            parent, length = tree.parent(node)
            path.append((node, length))
            node = parent
        path.append((node, 0))  # Add root with distance 0
        return path[::-1]  # Reverse to get root-to-node path

    # Get paths from root to both leaves
    path1 = path_to_root(leaf1)
    path2 = path_to_root(leaf2)

    # Find the lowest common ancestor (LCA)
    lca_index = 0
    while (
        lca_index < len(path1)
        and lca_index < len(path2)
        and path1[lca_index][0] == path2[lca_index][0]
    ):
        lca_index += 1
    lca_index -= 1

    # Calculate distances from LCA to each leaf
    distance1 = sum(length for _, length in path1[lca_index + 1 :])
    distance2 = sum(length for _, length in path2[lca_index + 1 :])

    # Return the total distance
    return distance1 + distance2


def _test_get_leaf_distance():
    # Initialize the first tree
    tree1 = Tree()
    tree1.add_nodes(["A", "B", "C", "D", "E"])
    tree1.add_edges(
        [("A", "B", 1.0), ("B", "C", 2.0), ("B", "D", 1.0), ("A", "E", 4.0)]
    )

    # Initialize the second tree with a different topology
    tree2 = Tree()
    tree2.add_nodes(["A", "B", "C", "D", "E"])
    tree2.add_edges(
        [("A", "C", 3.0), ("A", "B", 2.0), ("B", "D", 1.0), ("B", "E", 1.0)]
    )

    assert get_leaf_distance(tree1, leaf1="C", leaf2="D") == 3
    assert get_leaf_distance(tree1, leaf1="C", leaf2="E") == 7
    assert get_leaf_distance(tree1, leaf1="D", leaf2="E") == 6
    assert get_leaf_distance(tree2, leaf1="C", leaf2="D") == 6
    assert get_leaf_distance(tree2, leaf1="C", leaf2="E") == 6
    assert get_leaf_distance(tree2, leaf1="D", leaf2="E") == 2


_test_get_leaf_distance()


def filter_transitions_by_protein_and_length(
    transitions, which_protein: str = "y", max_length: int = 1022
):
    # lls have all transitions where one of the sequences are too long filtered out
    if which_protein == "x":
        transitions = [
            (x1_aln, x2_aln, tx)
            for x1_aln, y1_aln, x2_aln, y2_aln, tx, ty in transitions
            if len(x1_aln) <= max_length
            and len(x2_aln) <= max_length
            and len(y1_aln) <= max_length
            and len(y2_aln) <= max_length
        ]
    else:
        transitions = [
            (y1_aln, y2_aln, ty)
            for x1_aln, y1_aln, x2_aln, y2_aln, tx, ty in transitions
            if len(x1_aln) <= max_length
            and len(x2_aln) <= max_length
            and len(y1_aln) <= max_length
            and len(y2_aln) <= max_length
        ]
    return transitions
