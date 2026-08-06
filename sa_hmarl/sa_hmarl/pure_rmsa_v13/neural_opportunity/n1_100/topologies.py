"""Additional topologies for the N1-100 XLRON-consistent experiments.

The project registry already contains ``cost239_deeprmsa``.  ``abilene`` is not
part of the shipped registry, so we register it here from the canonical XLRON
topology file so that the new-topology experiments can use the same
``load_topology`` / ``make_env`` path as the existing ones.
"""
from __future__ import annotations

import sa_hmarl.network.topology_data as _topology_data

# Abilene: 12 nodes, 15 undirected links, distances in km.
# Source: https://raw.githubusercontent.com/micdoh/XLRON/main/xlron/data/topologies/abilene_undirected.json
# Node IDs in the JSON are 1-based; converted to 0-based below.
ABILENE_EDGES = [
    (0, 1, 199.0),
    (1, 4, 1500.0),
    (1, 5, 885.0),
    (1, 11, 1349.0),
    (2, 5, 389.0),
    (2, 8, 1500.0),
    (3, 6, 1116.0),
    (3, 9, 1892.0),
    (3, 10, 1964.0),
    (4, 6, 1500.0),
    (4, 7, 2741.0),
    (5, 6, 1352.0),
    (7, 9, 755.0),
    (8, 11, 502.0),
    (9, 10, 1500.0),
]


def register_extra_topologies() -> None:
    """Add non-registry topologies used by this experiment."""
    _topology_data.TOPOLOGY_REGISTRY["abilene"] = ABILENE_EDGES
