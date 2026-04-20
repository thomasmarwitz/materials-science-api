import argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from predict.graph import Graph


def build_index(graph_path: str, since: int, output_dir: str):
    graph = Graph.from_path(graph_path)
    edges = graph.get_until_year(since)

    if len(edges) == 0:
        raise ValueError("No edges found for selected cutoff year.")

    max_node = int(max(edges[:, 0].max(), edges[:, 1].max()))
    adj = Graph.build_adj_matrix(edges, binary=True, dim=max_node + 1).tocsr()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    vertices = np.unique(np.concatenate((edges[:, 0], edges[:, 1]))).astype(np.int64)
    degrees = np.diff(adj.indptr).astype(np.int32)
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int32)

    np.save(output / "vertices.npy", vertices)
    np.save(output / "degrees.npy", degrees)
    np.save(output / "indptr.npy", indptr)
    np.save(output / "indices.npy", indices)

    print(f"Wrote adjacency index to {output}")
    print(
        f"nodes={adj.shape[0]} vertices={len(vertices)} edges_undirected={len(indices) // 2}"
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="Path to graph pickle")
    parser.add_argument("--since", type=int, required=True, help="Cutoff year")
    parser.add_argument(
        "--output",
        default="data/adjacency",
        help="Output directory for adjacency index files",
    )
    args = parser.parse_args()

    build_index(args.graph, args.since, args.output)


if __name__ == "__main__":
    main()
