"""
Debug script to understand why GNNExplainer gives different results
between original_xai and parallel_xai.
"""
import torch
import numpy as np
from scipy.sparse import lil_matrix
from src.utils import set_seed, set_torch_config
from src.explain.utils import load_artifacts
from src.explain.methods import GNNExplainer
from src.explain.cluster import get_extended_data
from src.explain.sparse import normalize_sparse_matrix
from torch_geometric.data import Data

# Set seed
device = set_torch_config()

# Load data and model
data, model = load_artifacts("Cora", "gcn", device)
print(f"Data: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")

# Create explainer
explainer = GNNExplainer(model)

# Now let's trace step by step what happens for a specific node
# First, let's see what original_xai does for node 0
data_dev = data.to(device)

# Test: explain a single node on the full graph (like original_xai)
set_seed(42)
node_id = 5  # pick a node that has neighbors
fm_original = explainer.explain(data_dev.x, data_dev.edge_index, node_id)
print(f"\nNode {node_id} - Original explain on full graph:")
print(f"  Non-zero entries: {(fm_original != 0).sum().item()}")
print(f"  Non-zero indices: {torch.nonzero(fm_original).squeeze().tolist()}")
print(f"  Non-zero values: {fm_original[fm_original != 0].tolist()}")

# Now let's create the extended data with 2 clusters
num_clusters = 2
data_extended = get_extended_data(data_dev, num_clusters, 2, 0.0, device)

print(
    f"\nExtended data: {data_extended.x.shape[0]} nodes, {data_extended.edge_index.shape[1]} edges"
)
print(f"Cluster IDs unique: {data_extended.cluster_ids.unique().tolist()}")
print(
    f"Batch indexes range: [{data_extended.batch_indexes.min().item()}, {data_extended.batch_indexes.max().item()}]"
)

# Find node 5 in data_extended
node_positions = (data_extended.node_ids == node_id).nonzero().squeeze()
print(
    f"\nNode {node_id} appears at positions in data_extended: {node_positions.tolist()}"
)

# For each occurrence, check which cluster it belongs to and its batch_index
for pos in node_positions if node_positions.dim() > 0 else [node_positions]:
    pos = pos.item()
    cluster = data_extended.cluster_ids[pos].item()
    batch_idx = data_extended.batch_indexes[pos].item()
    print(f"  Position {pos}: cluster={cluster}, batch_index={batch_idx}")

# Now run explain on data_extended for a single batch that contains node 5
# Find which batch_index corresponds to node 5 being a target
target_positions = (data_extended.batch_indexes >= 0) & (
    data_extended.node_ids == node_id
)
if target_positions.any():
    target_pos = target_positions.nonzero().squeeze()
    print(
        f"\nNode {node_id} is a TARGET (batch_index >= 0) at position: {target_pos.tolist()}"
    )
    if target_pos.dim() == 0:
        target_pos = target_pos.unsqueeze(0)
    for tp in target_pos:
        tp = tp.item()
        batch_idx = data_extended.batch_indexes[tp].item()
        print(f"  Position {tp}: batch_index={batch_idx}")

        # Get all xai_ids for this batch_index (one per cluster)
        new_node_ids = torch.arange(data_extended.x.shape[0]).to(device)
        xai_ids = new_node_ids[data_extended.batch_indexes == batch_idx]
        print(f"  xai_ids for batch {batch_idx}: {xai_ids.tolist()}")
        print(
            f"  These correspond to original nodes: {data_extended.node_ids[xai_ids].tolist()}"
        )

        # Explain just this one node on the extended graph
        set_seed(42)
        fm_single = explainer.explain(
            data_extended.x, data_extended.edge_index, torch.tensor([tp]).to(device)
        )
        print(
            f"\n  Single-node explain on extended graph (just node {node_id} at pos {tp}):"
        )
        nz = (fm_single != 0).nonzero().squeeze()
        if nz.dim() == 0:
            nz = nz.unsqueeze(0)
        print(f"    Non-zero positions: {nz.tolist()[:20]}")
        print(
            f"    Original node IDs at those positions: {data_extended.node_ids[nz].tolist()[:20]}"
        )
        print(f"    Values: {fm_single[nz].tolist()[:20]}")

        # Now explain the full batch (all xai_ids)
        set_seed(42)
        fm_batch = explainer.explain(data_extended.x, data_extended.edge_index, xai_ids)
        print(
            f"\n  Batch explain on extended graph (batch {batch_idx}, {len(xai_ids)} nodes):"
        )
        nz_batch = (fm_batch != 0).nonzero().squeeze()
        if nz_batch.dim() == 0:
            nz_batch = nz_batch.unsqueeze(0)
        print(f"    Non-zero positions: {nz_batch.tolist()[:30]}")
        print(
            f"    Original node IDs at those positions: {data_extended.node_ids[nz_batch].tolist()[:30]}"
        )

        # Compare the mask values for node 5's neighborhood between single and batch
        # Get the positions in node 5's subgraph
        cluster_of_node5 = data_extended.cluster_ids[tp].item()
        node5_cluster_mask = data_extended.cluster_ids == cluster_of_node5
        node5_positions = node5_cluster_mask.nonzero().squeeze()

        print(
            f"\n  Comparison for node {node_id}'s cluster ({cluster_of_node5}) positions:"
        )
        print(
            f"    Single-node mask values: {fm_single[node5_positions].tolist()[:15]}"
        )
        print(f"    Batch mask values:       {fm_batch[node5_positions].tolist()[:15]}")
        diff = (fm_single[node5_positions] - fm_batch[node5_positions]).abs()
        print(f"    Max diff in this cluster: {diff.max().item()}")

        # Only check first occurrence
        break
