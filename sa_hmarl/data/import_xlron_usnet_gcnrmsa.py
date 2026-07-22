#!/usr/bin/env python3
"""Fetch and convert XLRON GCN-RMSA USNET topology to SA-HMARL edge list.

Source: https://github.com/micdoh/XLRON
File: xlron/data/topologies/usnet_gcnrnn_undirected.json
Commit: d07980b3233b1edc93507f60dc4a1c64b37af2e9 (latest touching this file)

This script downloads the official XLRON undirected topology, converts
1-based node IDs to 0-based, and writes the edge list in SA-HMARL format.
The conversion is deterministic and reproducible.
"""
import json
import hashlib
import urllib.request
from pathlib import Path
from typing import List, Tuple

XLRON_COMMIT = "d07980b3233b1edc93507f60dc4a1c64b37af2e9"
XLRON_URL = (
    "https://raw.githubusercontent.com/micdoh/XLRON/"
    f"{XLRON_COMMIT}/xlron/data/topologies/usnet_gcnrnn_undirected.json"
)
SOURCE_URL = (
    "https://github.com/micdoh/XLRON/blob/"
    f"{XLRON_COMMIT}/xlron/data/topologies/usnet_gcnrnn_undirected.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_xlron_usnet(raw_path: Path) -> str:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(XLRON_URL, raw_path)
    return sha256_file(raw_path)


def convert_to_edge_list(raw_path: Path) -> List[Tuple[int, int, float]]:
    with open(raw_path) as f:
        data = json.load(f)
    nodes = sorted({n["id"] for n in data["nodes"]})
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    assert len(nodes) == 24, f"Expected 24 nodes, got {len(nodes)}"
    edges = []
    for link in data["links"]:
        u = node_to_idx[link["source"]]
        v = node_to_idx[link["target"]]
        d = float(link["distance"])
        edges.append((u, v, d))
    assert len(edges) == 43, f"Expected 43 undirected links, got {len(edges)}"
    return edges


def main():
    output_dir = Path("/mnt/d/project/DNN_Agent/sa_hmarl/data/topologies")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "usnet_gcnrnn_undirected.json"

    raw_sha256 = fetch_xlron_usnet(raw_path)
    edges = convert_to_edge_list(raw_path)

    edge_list_path = output_dir / "usnet_gcnrnn_edge_list.json"
    with open(edge_list_path, "w") as f:
        json.dump({
            "topology": "xlron_usnet_gcnrmsa",
            "source_url": SOURCE_URL,
            "xlron_commit": XLRON_COMMIT,
            "raw_file_sha256": raw_sha256,
            "conversion_script": Path(__file__).name,
            "num_nodes": 24,
            "num_undirected_links": 43,
            "edges": edges,
        }, f, indent=2)

    print(f"Downloaded raw file to {raw_path}")
    print(f"Raw SHA-256: {raw_sha256}")
    print(f"Wrote edge list to {edge_list_path}")
    print(f"Topology: 24 nodes, {len(edges)} undirected links")


if __name__ == "__main__":
    main()
