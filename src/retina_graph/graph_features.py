from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import dilation, erosion, disk, skeletonize


def perturbation_set(mask: np.ndarray, fov: np.ndarray) -> dict[str, np.ndarray]:
    vessel = np.asarray(mask, dtype=bool) & np.asarray(fov, dtype=bool)
    return {
        "original": vessel,
        "erosion_r1": erosion(vessel, footprint=disk(1)) & fov,
        "erosion_r2": erosion(vessel, footprint=disk(2)) & fov,
        "dilation_r1": dilation(vessel, footprint=disk(1)) & fov,
    }


def _pixel_graph(skeleton: np.ndarray):
    coords = np.argwhere(skeleton)
    count = len(coords)
    if count == 0:
        return coords, csr_matrix((0, 0), dtype=np.float32), np.empty(0, dtype=np.int32)
    index_map = np.full(skeleton.shape, -1, dtype=np.int32)
    index_map[tuple(coords.T)] = np.arange(count, dtype=np.int32)
    rows, cols, weights = [], [], []
    height, width = skeleton.shape
    for dy, dx, weight in ((0, 1, 1.0), (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        source = index_map[y0:y1, x0:x1]
        target = index_map[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        valid = (source >= 0) & (target >= 0)
        if np.any(valid):
            u = source[valid]
            v = target[valid]
            rows.extend(u.tolist())
            cols.extend(v.tolist())
            weights.extend([weight] * len(u))
    row = np.asarray(rows, dtype=np.int32)
    col = np.asarray(cols, dtype=np.int32)
    weight = np.asarray(weights, dtype=np.float32)
    graph = csr_matrix((np.concatenate((weight, weight)), (np.concatenate((row, col)), np.concatenate((col, row)))), shape=(count, count))
    return coords, graph, np.diff(graph.indptr).astype(np.int32)


def _branch_record(path, length, coords, diameter_map):
    points = coords[np.asarray(path, dtype=np.int32)].astype(np.float64)
    displacement = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    tortuosity = float(length / displacement) if displacement > 0 else float("nan")
    if len(points) >= 5:
        smooth = gaussian_filter1d(points, sigma=1.0, axis=0, mode="nearest")
        smooth[0], smooth[-1] = points[0], points[-1]
        smooth_length = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
        smooth_tortuosity = float(smooth_length / displacement) if displacement > 0 else float("nan")
    else:
        smooth_tortuosity = tortuosity
    pixel_coords = coords[np.asarray(path, dtype=np.int32)]
    diameters = diameter_map[tuple(pixel_coords.T)]
    return {
        "length_px": float(length),
        "tortuosity": tortuosity,
        "smoothed_tortuosity": smooth_tortuosity,
        "mean_diameter_px": float(np.mean(diameters)) if len(diameters) else float("nan"),
        "median_diameter_px": float(np.median(diameters)) if len(diameters) else float("nan"),
    }


def _summarize_graph(node_types, node_positions, branches, prune_length_px):
    graph = nx.MultiGraph()
    for node_id, node_type in node_types.items():
        y, x = node_positions[node_id]
        graph.add_node(node_id, node_type=node_type, y=float(y), x=float(x))
    for branch in branches:
        graph.add_edge(
            branch["source"],
            branch["target"],
            weight=branch["length_px"],
            length_px=branch["length_px"],
            tortuosity=branch["tortuosity"] if np.isfinite(branch["tortuosity"]) else -1.0,
            median_diameter_px=branch["median_diameter_px"] if np.isfinite(branch["median_diameter_px"]) else -1.0,
        )
    degree = dict(graph.degree())
    tortuosities = [b["tortuosity"] for b in branches if np.isfinite(b["tortuosity"])]
    smooth_tortuosities = [b["smoothed_tortuosity"] for b in branches if np.isfinite(b["smoothed_tortuosity"])]
    diameters = [b["median_diameter_px"] for b in branches if np.isfinite(b["median_diameter_px"])]
    full = {
        "graph_node_count": float(graph.number_of_nodes()),
        "branch_count": float(len(branches)),
        "endpoint_count": float(sum(value == 1 for value in degree.values())),
        "junction_count": float(sum(value >= 3 for value in degree.values())),
        "graph_component_count": float(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0.0,
        "total_branch_length_px": float(sum(b["length_px"] for b in branches)),
        "mean_branch_length_px": float(np.mean([b["length_px"] for b in branches])) if branches else float("nan"),
        "mean_tortuosity": float(np.mean(tortuosities)) if tortuosities else float("nan"),
        "mean_smoothed_tortuosity": float(np.mean(smooth_tortuosities)) if smooth_tortuosities else float("nan"),
        "median_vessel_diameter_px": float(np.median(diameters)) if diameters else float("nan"),
    }
    kept = [b for b in branches if b["length_px"] >= prune_length_px]
    pruned_graph = nx.MultiGraph()
    for b in kept:
        pruned_graph.add_edge(b["source"], b["target"], weight=b["length_px"])
    pruned_degree = dict(pruned_graph.degree())
    full.update({
        "pruned_branch_count": float(len(kept)),
        "pruned_total_length_px": float(sum(b["length_px"] for b in kept)),
        "pruned_endpoint_count": float(sum(value == 1 for value in pruned_degree.values())),
        "pruned_junction_count": float(sum(value >= 3 for value in pruned_degree.values())),
        "pruned_component_count": float(nx.number_connected_components(pruned_graph)) if pruned_graph.number_of_nodes() else 0.0,
    })
    return full, graph


def extract_graph_features(vessel_mask: np.ndarray, fov_mask: np.ndarray, prune_length_px: float = 5.0):
    fov = np.asarray(fov_mask, dtype=bool)
    vessel = np.asarray(vessel_mask, dtype=bool) & fov
    skeleton = skeletonize(vessel)
    coords, adjacency, degrees = _pixel_graph(skeleton)
    if len(coords) == 0:
        metrics = _empty_metrics()
        return metrics, nx.MultiGraph(), []

    junction_pixels = np.zeros(skeleton.shape, dtype=bool)
    junction_indices = np.flatnonzero(degrees >= 3)
    if len(junction_indices):
        junction_pixels[tuple(coords[junction_indices].T)] = True
    junction_labels, junction_count = ndi.label(junction_pixels, structure=np.ones((3, 3), dtype=np.uint8))

    node_for_pixel = np.full(len(coords), -1, dtype=np.int32)
    node_types = {}
    node_positions = {}
    next_node = 0
    for index in np.flatnonzero(degrees == 1):
        node_for_pixel[index] = next_node
        node_types[next_node] = "endpoint"
        node_positions[next_node] = tuple(coords[index])
        next_node += 1
    for label_id in range(1, junction_count + 1):
        members = np.flatnonzero(junction_labels[tuple(coords.T)] == label_id)
        node_for_pixel[members] = next_node
        node_types[next_node] = "junction"
        node_positions[next_node] = tuple(coords[members].mean(axis=0))
        next_node += 1
    for index in np.flatnonzero(degrees == 0):
        node_for_pixel[index] = next_node
        node_types[next_node] = "isolated"
        node_positions[next_node] = tuple(coords[index])
        next_node += 1

    diameter_map = 2.0 * distance_transform_edt(vessel)
    visited = set()
    branches = []
    n = len(coords)

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def add_branch(path, length, source, target):
        if not path:
            return
        values = _branch_record(path, length, coords, diameter_map)
        values.update({"source": int(source), "target": int(target)})
        branches.append(values)

    for start in np.flatnonzero(degrees != 2):
        source_node = int(node_for_pixel[start])
        if source_node < 0:
            continue
        for position in range(adjacency.indptr[start], adjacency.indptr[start + 1]):
            current = int(adjacency.indices[position])
            key = edge_key(int(start), current)
            if key in visited:
                continue
            visited.add(key)
            path = [int(start), current]
            length = float(adjacency.data[position])
            previous = int(start)
            while degrees[current] == 2:
                begin, end = adjacency.indptr[current], adjacency.indptr[current + 1]
                candidates = [int(v) for v in adjacency.indices[begin:end] if int(v) != previous]
                if not candidates:
                    break
                nxt = candidates[0]
                next_key = edge_key(current, nxt)
                if next_key in visited:
                    break
                edge_pos = next(i for i in range(begin, end) if int(adjacency.indices[i]) == nxt)
                length += float(adjacency.data[edge_pos])
                visited.add(next_key)
                path.append(nxt)
                previous, current = current, nxt
            target_node = int(node_for_pixel[current]) if node_for_pixel[current] >= 0 else source_node
            if source_node != target_node or len(path) > 2:
                add_branch(path, length, source_node, target_node)

    for start in range(n):
        begin, end = adjacency.indptr[start], adjacency.indptr[start + 1]
        for position in range(begin, end):
            current = int(adjacency.indices[position])
            key = edge_key(start, current)
            if key in visited:
                continue
            visited.add(key)
            path = [start, current]
            length = float(adjacency.data[position])
            previous = start
            while current != start:
                begin2, end2 = adjacency.indptr[current], adjacency.indptr[current + 1]
                candidates = [int(v) for v in adjacency.indices[begin2:end2] if int(v) != previous]
                if not candidates:
                    break
                nxt = candidates[0]
                next_key = edge_key(current, nxt)
                if next_key in visited:
                    break
                edge_pos = next(i for i in range(begin2, end2) if int(adjacency.indices[i]) == nxt)
                length += float(adjacency.data[edge_pos])
                visited.add(next_key)
                path.append(nxt)
                previous, current = current, nxt
            cycle_node = next_node
            node_types[cycle_node] = "cycle"
            node_positions[cycle_node] = tuple(coords[np.asarray(path, dtype=np.int32)].mean(axis=0))
            next_node += 1
            add_branch(path, length, cycle_node, cycle_node)

    metrics, graph = _summarize_graph(node_types, node_positions, branches, prune_length_px)
    metrics["skeleton_pixel_count"] = float(len(coords))
    metrics["skeleton_component_count"] = float(ndi.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))[1])
    metrics["vessel_area_pixels"] = float(np.count_nonzero(vessel))
    metrics["vessel_area_fraction"] = float(np.count_nonzero(vessel) / np.count_nonzero(fov)) if np.count_nonzero(fov) else float("nan")
    return metrics, graph, branches


def _empty_metrics():
    return {
        "graph_node_count": 0.0, "branch_count": 0.0, "endpoint_count": 0.0,
        "junction_count": 0.0, "graph_component_count": 0.0,
        "total_branch_length_px": 0.0, "mean_branch_length_px": float("nan"),
        "mean_tortuosity": float("nan"), "mean_smoothed_tortuosity": float("nan"),
        "median_vessel_diameter_px": float("nan"), "pruned_branch_count": 0.0,
        "pruned_total_length_px": 0.0, "pruned_endpoint_count": 0.0,
        "pruned_junction_count": 0.0, "pruned_component_count": 0.0,
        "skeleton_pixel_count": 0.0, "skeleton_component_count": 0.0,
        "vessel_area_pixels": 0.0, "vessel_area_fraction": float("nan"),
    }
