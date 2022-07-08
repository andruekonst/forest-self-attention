import numpy as np


def _prepare_leaf_data_fast(xs, y, leaf_ids, estimators):
    """Utility function for preparing forest tree leaf data.
    For each leaf tree finds all corresponding training samples,
    and calculates averages for input vectors and target values.

    Args:
        xs: Input vectors.
        y: Target values.
        leaf_ids: Input samples to leaves ids correspondence
                  (see `sklearn.ensemble.RandomForestClassifier.apply`).
        estimators: List of estimators.

    Returns:
        A pair of leaf data for input vectors and target values.

    """
    max_leaf_id = max(map(lambda e: e.tree_.node_count, estimators))
    y_len = 1 if y.ndim == 1 else y.shape[1]
    result_x = np.full((len(estimators), max_leaf_id + 1, xs.shape[1]), np.nan, dtype=np.float32)
    result_y = np.full((len(estimators), max_leaf_id + 1, y_len), np.nan, dtype=np.float32)
    for tree_id in range(len(estimators)):
        for leaf_id in range(estimators[tree_id].tree_.node_count + 1):
            mask = (leaf_ids[:, tree_id] == leaf_id)
            masked_xs = xs[mask]
            masked_y = y[mask]
            if mask.any():
                result_x[tree_id, leaf_id] = masked_xs.mean(axis=0)
                result_y[tree_id, leaf_id] = masked_y.mean(axis=0)
    return result_x, result_y
