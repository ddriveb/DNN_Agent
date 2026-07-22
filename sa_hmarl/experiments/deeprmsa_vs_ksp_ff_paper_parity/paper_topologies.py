"""Paper-standard topology data local to this pipeline.

COST239_DEEPRMSA: the COST239 variant used by upstream DeepRMSA
(``DeepRMSA/Cost_239.m``) and shipped by XLRON as
``cost239_deeprmsa_undirected.json`` — distances are 2x the great-circle
("real") values.  Node IDs converted to 0-based.  This is the COST239 the
paper's DeepRMSA comparison refers to; the repository's
``xlron_cost239_ptrnet_real`` is a different COST239 variant (PtrNet real
distances) and is NOT used here (see PREFLIGHT_AUDIT.md, addendum).
"""
from typing import Dict, List, Tuple

# (u, v, length_km), 0-based, 11 nodes, 26 undirected links.
COST239_DEEPRMSA_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 900.0),
    (0, 2, 780.0),
    (0, 3, 1100.0),
    (0, 7, 2620.0),
    (1, 2, 600.0),
    (1, 4, 800.0),
    (1, 5, 1200.0),
    (1, 6, 1640.0),
    (1, 8, 2180.0),
    (2, 3, 420.0),
    (2, 4, 440.0),
    (2, 6, 1860.0),
    (3, 4, 780.0),
    (3, 7, 1520.0),
    (3, 8, 1320.0),
    (4, 5, 700.0),
    (4, 9, 1460.0),
    (5, 6, 640.0),
    (5, 9, 1130.0),
    (5, 10, 1460.0),
    (6, 10, 1640.0),
    (7, 8, 780.0),
    (7, 9, 1480.0),
    (8, 9, 680.0),
    (8, 10, 1320.0),
    (9, 10, 640.0),
]

LOCAL_TOPOLOGIES: Dict[str, List[Tuple[int, int, float]]] = {
    "cost239_deeprmsa": COST239_DEEPRMSA_EDGES,
}
