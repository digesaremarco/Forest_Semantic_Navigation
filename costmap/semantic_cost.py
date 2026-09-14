"""
Semantic traversability cost.

Computes C_sem(i,j), the semantic component of the traversability
cost, as the expected per-class cost under the cell's class
probability distribution:

    C_sem(i,j) = sum_k p_{i,j,k} * c_class(k)

Cost convention: [0, 1], 0 = fully traversable, 1 = maximally
costly. A cell is NaN wherever it was never observed (its own
probability vector is all-NaN), consistent with GeometricCost and
with GridMap's own NaN convention -- no extra masking is needed for
this: NaN propagates on its own through the weighted sum.
"""

import numpy as np


class SemanticCost:

    def __init__(self, costmap_config, class_reducer):
        self.costmap_config = costmap_config
        self.class_reducer = class_reducer

        # Same channel order GridMap uses for semantic_probs / the
        # semantic_alpha layer: ClassReducer.get_kept_classes().
        self.class_names = [
            self.class_reducer.get_class_name(class_id)
            for class_id in self.class_reducer.get_kept_classes()
        ]

        self.cost_vector = np.array(
            [
                self.costmap_config.get_cost(class_name)
                for class_name in self.class_names
            ],
            dtype=np.float32
        )

    def compute(self, semantic_probs):
        """
        Compute the semantic traversability cost.

        Parameters
        ----------
        semantic_probs : numpy.ndarray
            Shape (H, W, C), float32. Per-cell class probability
            distribution, in the same channel order as
            ClassReducer.get_kept_classes() -- i.e. what
            GridMap.get_semantic_probs_layer() returns. NaN for
            unobserved cells.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), float32, in [0, 1]. NaN for unobserved
            cells.
        """

        self.validate_semantic_probs(semantic_probs)

        cost = np.tensordot(
            semantic_probs, self.cost_vector, axes=([2], [0])
        )

        return cost.astype(np.float32)

    def get_class_cost(self, class_name):
        """Return the configured cost for a single class, by name."""

        return self.costmap_config.get_cost(class_name)

    def validate_semantic_probs(self, semantic_probs):
        if (
            not isinstance(semantic_probs, np.ndarray)
            or semantic_probs.ndim != 3
        ):
            raise ValueError(
                "semantic_probs must have shape (H, W, C), got "
                f"{getattr(semantic_probs, 'shape', type(semantic_probs))}."
            )

        if semantic_probs.shape[2] != len(self.class_names):
            raise ValueError(
                f"semantic_probs must have {len(self.class_names)} "
                "channels (one per retained class), got "
                f"{semantic_probs.shape[2]}."
            )