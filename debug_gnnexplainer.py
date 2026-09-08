"""
Debug script to understand why GNNExplainer gives different results
between original_xai and parallel_xai.
"""
import torch
from src.utils import set_seed, set_torch_config
from src.explain.utils import load_artifacts, METHODS
from src.explain.methods import GNNExplainer
from src.explain.executions import original_xai, parallel_xai

# Set seed
device = set_torch_config()

# Load data and model
data, model = load_artifacts("Cora", "gcn", device)
print(f"Data: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")

# Create explainer
explainer = GNNExplainer(model)

# Run original
print("\n--- Running original_xai ---")
set_seed(42)
original_result = original_xai(explainer, data, device=device)

# Run parallel with 2 clusters
print("\n--- Running parallel_xai (2 clusters) ---")
set_seed(42)
parallel_result, _, _ = parallel_xai(explainer, data, 2, 0.0, device=device)

# Compare
from scipy.sparse import lil_matrix
import numpy as np

orig_dense = original_result.toarray()
par_dense = parallel_result.toarray()

print(f"\nOriginal non-zero: {np.count_nonzero(orig_dense)}")
print(f"Parallel non-zero: {np.count_nonzero(par_dense)}")

# Check if they're the same
diff = np.abs(orig_dense - par_dense)
print(f"Max difference: {diff.max()}")
print(
    f"Mean difference (where non-zero): {diff[diff > 0].mean() if np.any(diff > 0) else 0}"
)
print(f"Number of differing entries: {np.count_nonzero(diff)}")

# Check a specific node
node = 0
print(f"\nNode {node} original row (non-zero values):")
orig_row = orig_dense[node]
par_row = par_dense[node]
orig_nz = np.nonzero(orig_row)[0]
par_nz = np.nonzero(par_row)[0]
print(f"  Original non-zero indices: {orig_nz[:20]}")
print(f"  Parallel non-zero indices: {par_nz[:20]}")
if len(orig_nz) > 0:
    print(f"  Original values: {orig_row[orig_nz[:10]]}")
if len(par_nz) > 0:
    print(f"  Parallel values: {par_row[par_nz[:10]]}")
