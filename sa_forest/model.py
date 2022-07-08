import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple
from .forests import *
from scipy.special import softmax as _softmax
import cvxpy as cp
from .leaf_data import _prepare_leaf_data_fast
from .utils import _get_node_depths, _convert_labels_to_probas
from .solver import _try_solve_problem


class SAFParams(NamedTuple):
    """Parameters of Self-Attention Forest.
    """
    kind: ForestKind
    task: TaskType
    loss_ord: int = 2
    eps: float = 0.5
    tau: float = 1.0
    gamma: float = 1.0
    sa_tau: float = 1.0
    sa_zero_diagonals: bool = True
    sa_dist: str = 'y'
    forest: dict = {}


class BaseAttentionForest(ABC):
    """Base Attention Forest.
    Preprocesses inputs and trains an underlying forest.
    """
    def __init__(self, params):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _preprocess_target(self, y):
        """Preprocess target.
        Converts classification labels to probabilities.

        Args:
            y: Input labels.

        Returns:
            Encoded labels (probabilities).

        """
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y

    def fit(self, X, y) -> 'BaseAttentionForest':
        """Fit an underlying forest and obtain leaf data.

        Args:
            X: Input vectors.
            y: Input targets.

        Returns:
            Self.

        """
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.fit(X, y)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        # collect leaf data
        if hasattr(self.forest, 'get_leaf_data'):
            self.leaf_data_x, self.leaf_data_y = self.forest.get_leaf_data()
        else:
            self.leaf_data_x, self.leaf_data_y = _prepare_leaf_data_fast(
                self.training_xs,
                self.training_y,
                self.training_leaf_ids,
                self.forest.estimators_,
            )
        self.tree_weights = np.ones(self.forest.n_estimators)
        self.static_weights = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        return self

    @abstractmethod
    def optimize_weights(self, X, y_orig) -> 'BaseAttentionForest':
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    def predict_original(self, X):
        """Predict with the underlying forest.

        Args:
            X: Input vectors.

        Returns:
            Predictions.

        """
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')


class SelfAttentionForest(BaseAttentionForest):
    """Self-Attention Forest
    """
    def __init__(self, params: SAFParams):
        self.params = params
        self.forest = None
        self.w = None
        self.v = None
        self._after_init()

    def fit(self, X, y):
        super().fit(X, y)
        self.w = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        self.v = np.ones(self.forest.n_estimators) / self.forest.n_estimators

    def optimize_weights(self, X, y_orig) -> 'SelfAttentionForest':
        """Estimate optimal weights values based on the given data set.

        Args:
            X: Input vectors.
            y_orig: Input targets.

        Returns:
            Self.

        """
        assert self.forest is not None, "Need to fit before weights optimization"
        dynamic_weights, dynamic_x, dynamic_y = self._get_dynamic_weights_y(X)
        static_weights = cp.Variable((1, self.forest.n_estimators))
        v_weights = cp.Variable((1, self.forest.n_estimators))

        bias = 0.0
        y = y_orig.copy()
        y -= bias

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights
        else:
            # y0: y0t0_0 y0t0_1 ... y0t0_d | y0t1_0 y0t1_1 ... y0t1_d | ... | y0tT_0 ... y0tT_d
            # loss = sum_i sum_j (sum_k yitk_j - yi_j)
            # dynamic_y = dynamic_y.reshape((dynamic_y.shape[0], -1))
            # swap 1 and 2 axes and merge "sample" and "feature" axes
            n_trees = dynamic_y.shape[1]
            n_outs = dynamic_y.shape[2]
            dynamic_y = np.transpose(dynamic_y, (0, 2, 1)).reshape((-1, dynamic_y.shape[1]))
            y = y.reshape((-1))
            # dynamic_weights shape: (n_samples, n_trees)
            # repeat dynamic weights for each output
            dynamic_weights = np.tile(dynamic_weights[:, np.newaxis, :], (1, n_outs, 1)).reshape((-1, n_trees))
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
            y = y.toarray().ravel()
            # print("Shapes:", mixed_weights.shape, dynamic_y.shape)


        # self-attention
        sa_softmax = self._self_attention_softmax(dynamic_x, dynamic_y)
        # sa_weights = (1.0 - self.params.gamma) * sa_softmax + self.params.gamma * self.v[:, np.newaxis]
        sa_softmax_y = np.einsum('ijk,ik->ij', (1.0 - self.params.gamma) * sa_softmax, dynamic_y)
        # sa_softmax_y shape: (n_points, n_trees)
        sa_static_y = self.params.gamma * cp.sum(cp.multiply(v_weights, dynamic_y), axis=1)

        mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights

        target_approx = cp.sum(cp.multiply(mixed_weights, sa_softmax_y), axis=1) + sa_static_y
        loss_terms = target_approx - y
        if self.params.loss_ord == 1:
            min_obj = cp.sum(cp.abs(loss_terms))
        elif self.params.loss_ord == 2:
            min_obj = cp.sum_squares(loss_terms)
        else:
            raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
        problem = cp.Problem(cp.Minimize(min_obj),
            [
                static_weights >= 0,
                cp.sum(static_weights, axis=1) == 1,
                v_weights >= 0,
                cp.sum(v_weights, axis=1) == 1,
            ]
        )

        loss_value = _try_solve_problem(problem, static_weights)

        if static_weights.value is None:
            logging.warn(f"Weights optimization error (eps={self.params.eps}). Using default values.")
        else:
            self.w = static_weights.value.copy().reshape((-1,))
            self.v = v_weights.value.copy().reshape((-1,))
        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        all_x = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            tree_dynamic_x = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                # cur_data = self.leaf_data[cur_tree_id][cur_leaf_id]
                # leaf_mean_x = cur_data.xs.mean(axis=0)
                # leaf_mean_y = cur_data.y.mean(axis=0)
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
                tree_dynamic_x.append(leaf_mean_x)
            tree_dynamic_weights = _softmax(np.array(tree_dynamic_weights) * self.params.tau)
            tree_dynamic_y = np.array(tree_dynamic_y)
            tree_dynamic_x = np.array(tree_dynamic_x)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_x.append(tree_dynamic_x)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_x, all_y

    def _self_attention_softmax(self, all_x, all_y):
        """Estimate Self-Attention softmax values.

        Args:
            all_x: Input vectors.
            all_y: Input targets.

        Returns:
            Self-attention softmax weights.

        """
        if self.params.sa_dist == 'y':
            neg_dist = -(all_y[:, np.newaxis] - all_y[..., np.newaxis]) ** 2
        elif self.params.sa_dist == 'x':
            neg_dist = -np.linalg.norm(all_x[:, np.newaxis] - all_x[..., np.newaxis, :], ord=2, axis=-1) ** 2
            if all_y.ndim == 3:
                neg_dist = neg_dist[..., np.newaxis]
        elif self.params.sa_dist == 'y/x':
            neg_dist_y = -(all_y[:, np.newaxis] - all_y[..., np.newaxis]) ** 2
            neg_dist = -np.linalg.norm(all_x[:, np.newaxis] - all_x[..., np.newaxis, :], ord=2, axis=-1) ** 2
            if all_y.ndim == 3:
                neg_dist = neg_dist[..., np.newaxis]
            neg_dist = neg_dist_y / np.maximum(neg_dist, 1.e-9)
        else:
            raise ValueError(f'Wrong sa_dist parameter value: "{self.params.sa_dist}"')
        # fill diagonals with -inf to obtain zero weights after softmax
        if self.params.sa_zero_diagonals:
            for i in range(neg_dist.shape[1]):
                neg_dist[:, i, i] = -np.inf
        sa_softmax = _softmax(self.params.sa_tau * neg_dist, axis=2)
        return sa_softmax

    def predict(self, X) -> np.ndarray:
        """Predict using the optimized weights.

        Args:
            X: Input vectors.

        Returns:
            Predictions.

        """
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_x, all_y = self._get_dynamic_weights_y(X)
        mixed_weights = (1.0 - self.params.eps) * all_dynamic_weights + self.params.eps * self.w
        mixed_weights = mixed_weights[..., np.newaxis]
        
        # self-attention
        sa_softmax = self._self_attention_softmax(all_x, all_y)
        sa_weights = (1.0 - self.params.gamma) * sa_softmax + self.params.gamma * self.v[:, np.newaxis]
        sa_y = np.einsum('ijkl,ikl->ijl', sa_weights, all_y)
        predictions = np.sum(mixed_weights * sa_y, axis=1)
        return predictions

