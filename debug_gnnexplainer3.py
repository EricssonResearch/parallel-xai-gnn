"""
Debug script - find exactly which nodes differ and why.
"""
import torch
import numpy as np
from scipy.sparse import lil_matrix
from src.utils import set_seed, set_torch_config
from src.explain.utils import load_artifacts
from src.explain.methods import GNNExplainer
from src.explain.executions import original_xai, parallel_xai
from src.explain.cluster import get_extended_data
from torch_geometric.data import Data

# Set seed
device = set_torch_config()

# Load data and model
data, model = load_artifacts("Cora", "gcn", device)
print(f"Data: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")

# Create explainer
explainer = GNNExplainer(model)

# Run original
set_seed(42)
original_result = original_xai(explainer, data, device=device)

# Run parallel
set_seed(42)
parallel_result, _, _ = parallel_xai(explainer, data, 2, 0.0, device=device)

# Find differing rows
orig_dense = original_result.toarray()
par_dense = parallel_result.toarray()
diff = np.abs(orig_dense - par_dense)

# Which rows (source nodes) differ?
differing_rows = np.where(diff.sum(axis=1) > 0)[0]
print(f"\nDiffering source nodes: {differing_rows.tolist()}")

# For each differing row, show details
for row in differing_rows[:5]:
    print(f"\nNode {row}:")
    orig_nz = np.nonzero(orig_dense[row])[0]
    par_nz = np.nonzero(par_dense[row])[0]
    print(
        f"  Original non-zero at: {orig_nz.tolist()}, values: {orig_dense[row][orig_nz].tolist()}"
    )
    print(
        f"  Parallel non-zero at: {par_nz.tolist()}, values: {par_dense[row][par_nz].tolist()}"
    )

# Now let's understand data_extended structure
data_dev = data.to(device)
data_extended = get_extended_data(data_dev, 2, 2, 0.0, device)

print(f"\n\n--- Extended data structure ---")
print(f"Extended nodes: {data_extended.x.shape[0]}, Original nodes: {data.x.shape[0]}")
print(
    f"Extended edges: {data_extended.edge_index.shape[1]}, Original edges: {data.edge_index.shape[1]}"
)

# Check: how many unique nodes in extended vs original
unique_node_ids = data_extended.node_ids.unique()
print(f"Unique original node IDs in extended: {len(unique_node_ids)}")

# Check node duplication
from collections import Counter

node_counts = Counter(data_extended.node_ids.tolist())
duplicated = {k: v for k, v in node_counts.items() if v > 1}
print(f"Duplicated nodes: {duplicated}")

# For each differing row node, check its cluster assignment and position
print("\n--- Checking differing nodes in extended data ---")
for row in differing_rows[:5]:
    positions = (data_extended.node_ids == row).nonzero().squeeze()
    if positions.dim() == 0:
        positions = positions.unsqueeze(0)
    for pos in positions:
        pos = pos.item()
        cluster = data_extended.cluster_ids[pos].item()
        batch_idx = data_extended.batch_indexes[pos].item()
        print(f"  Node {row}: pos={pos}, cluster={cluster}, batch_idx={batch_idx}")

# Let's also check: for a differing node, what does single-node explain give
# on the original graph vs on the extended graph
print("\n--- Detailed comparison for differing nodes ---")
for row in differing_rows[:3]:
    print(f"\nNode {row}:")

    # Original graph - single node explain
    set_seed(42)
    fm_orig = explainer.explain(data_dev.x, data_dev.edge_index, row)
    print(
        f"  On original graph: non-zero at {torch.nonzero(fm_orig).squeeze().tolist()}"
    )
    print(f"  Values: {fm_orig[fm_orig != 0].tolist()}")

    # Extended graph - find this node as target
    target_mask = (data_extended.batch_indexes >= 0) & (data_extended.node_ids == row)
    if target_mask.any():
        target_pos = target_mask.nonzero().squeeze().item()
        batch_idx = data_extended.batch_indexes[target_pos].item()

        # Single node on extended graph
        set_seed(42)
        fm_ext_single = explainer.explain(
            data_extended.x,
            data_extended.edge_index,
            torch.tensor([target_pos]).to(device),
        )
        nz = (fm_ext_single != 0).nonzero().squeeze()
        if nz.dim() == 0:
            nz = nz.unsqueeze(0)
        print(f"  On extended graph (single): non-zero at positions {nz.tolist()}")
        print(f"  Mapped to original IDs: {data_extended.node_ids[nz].tolist()}")
        print(f"  Values: {fm_ext_single[nz].tolist()}")
