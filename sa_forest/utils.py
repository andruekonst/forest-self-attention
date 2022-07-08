import numpy as np
from typing import Union
from pathlib import Path
import yaml, json
from sklearn.preprocessing import OneHotEncoder


def load_dict(file: Union[str, Path]):
    """Load a configuration dictionary from the file.

    Args:
        file: Input file with 'yml', 'yaml' or 'json' extension.

    Returns:
        Loaded dictionary.

    """
    path = Path(file) if isinstance(file, str) else file
    with open(file, 'r') as inf:
        if path.suffix in ['.yml', '.yaml']:
            data = yaml.safe_load(inf)
        elif path.suffix in ['.json']:
            data = json.load(inf)
        else:
            raise ValueError(f'Wrong dictionary type: "{path.suffix}"')
    return data


def _get_node_depths(tree):
    """Get the node depths of the decision tree

    >>> d = DecisionTreeClassifier()
    >>> d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    >>> _get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])

    Source: https://stackoverflow.com/questions/39476414/scikit-learn-decision-tree-node-depth

    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        depths += [current_depth]
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths) 
    return np.array(depths)


def _convert_labels_to_probas(y, encoder=None):
    """Convert classification labels to probabilities.

    Args:
        y: Input labels.
        encoder: One-hot encoder or None.

    Returns:
        A pair of encoded labels (probabilities) and one-hot encoder instance.

    """
    if y.ndim == 2 and y.shape[1] >= 2:
        return y, encoder
    if encoder is None:
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape((-1, 1)))
    else:
        y = encoder.transform(y.reshape((-1, 1)))
    return y, encoder

