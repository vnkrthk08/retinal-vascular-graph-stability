import numpy as np
from retina_graph.graph_features import extract_graph_features, perturbation_set


def test_graph_features_on_connected_branching_mask():
    mask = np.zeros((80, 80), dtype=bool)
    mask[8:72, 39:42] = True
    mask[25:29, 15:66] = True
    mask[50:54, 20:62] = True
    fov = np.ones_like(mask)
    metrics, graph, branches = extract_graph_features(mask, fov)
    assert metrics["skeleton_pixel_count"] > 0
    assert metrics["branch_count"] >= 1
    assert metrics["total_branch_length_px"] > 0
    assert graph.number_of_nodes() > 0
    assert len(branches) == int(metrics["branch_count"])


def test_graph_perturbations_are_clipped_to_fov():
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 18:22] = True
    fov = np.ones_like(mask)
    fov[:, :5] = False
    variants = perturbation_set(mask, fov)
    assert set(variants) == {"original", "erosion_r1", "erosion_r2", "dilation_r1"}
    assert all(value.shape == mask.shape for value in variants.values())
    assert all(not np.any(value & ~fov) for value in variants.values())
