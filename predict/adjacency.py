from pathlib import Path
import numpy as np


class AdjacencyIndex:
    REQUIRED_FILES = ("indptr.npy", "indices.npy", "degrees.npy", "vertices.npy")

    def __init__(self, index_dir: str | None):
        if not index_dir:
            raise ValueError(
                "Adjacency index path is not configured. "
                "Set ADJACENCY_INDEX or use default 'data/adjacency' and build it via: "
                "pixi run build-adjacency"
            )

        base = Path(index_dir)

        if not base.exists():
            raise FileNotFoundError(
                f"Adjacency index directory not found: {base}. "
                "Build it first with: pixi run build-adjacency"
            )

        missing = [name for name in self.REQUIRED_FILES if not (base / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Adjacency index is incomplete in {base}. Missing files: {', '.join(missing)}. "
                "Rebuild with: pixi run build-adjacency"
            )

        self.indptr = np.load(base / "indptr.npy", mmap_mode="r")
        self.indices = np.load(base / "indices.npy", mmap_mode="r")
        self.degrees = np.load(base / "degrees.npy", mmap_mode="r")
        self.vertices = np.load(base / "vertices.npy", mmap_mode="r")

    def degree(self, node_id: int) -> int:
        return int(self.degrees[node_id])

    def neighbors(self, node_id: int) -> np.ndarray:
        start = int(self.indptr[node_id])
        end = int(self.indptr[node_id + 1])
        return self.indices[start:end]

    def neighbor_set(self, node_id: int) -> set[int]:
        return set(self.neighbors(node_id).tolist())

    def has_edge(self, src: int, dst: int) -> bool:
        row = self.neighbors(src)
        if len(row) == 0:
            return False

        idx = np.searchsorted(row, dst)
        return idx < len(row) and int(row[idx]) == dst
