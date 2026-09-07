"""
Filters SegFormer semantic classes to retain only the classes
relevant for the navigation and traversability pipeline.
"""

import numpy as np


class ClassReducer:
    """Filters the original SegFormer semantic classes."""

    def __init__(self, config):
        """
        Initialize the class reducer.

        Args:
            config: PerceptionConfig instance.
        """
        self.config = config

        # Original SegFormer class IDs to keep.
        # The order defines the order of the output probability channels.
        self.keep_classes = [
            0,   # grass
            1,   # tree
            2,   # pole
            3,   # water
            5,   # vehicle
            6,   # container
            7,   # asphalt
            8,   # gravel
            9,   # mulch
            10,  # rockbed
            11,  # log
            15,  # bush
            17,  # rock
            19,  # concrete
            21,  # building
            23   # generic_ground
        ]

        # Mapping from the filtered channel index to the original SegFormer class ID.
        # Example: filtered channel 4 -> original class 5 (vehicle)
        self.filtered_class_ids = np.array(
            self.keep_classes,
            dtype=np.int32
        )

    def filter_probabilities(self, probabilities):
        """
        Keep only the probability channels corresponding to
        the selected semantic classes.

        Args:
            probabilities: numpy array with shape (C, H, W).

        Returns:
            Filtered probabilities with shape
            (num_kept_classes, H, W).
        """
        if not isinstance(probabilities, np.ndarray):
            raise TypeError(
                "Probabilities must be a numpy.ndarray."
            )

        if probabilities.ndim != 3:
            raise ValueError(
                f"Expected probabilities with 3 dimensions "
                f"(C, H, W), got shape {probabilities.shape}."
            )

        if probabilities.shape[0] != self.config.num_classes:
            raise ValueError(
                f"Expected {self.config.num_classes} classes, "
                f"got {probabilities.shape[0]}."
            )

        return probabilities[self.filtered_class_ids]

    def filter_mask(self, mask):
        """
        Filter a segmentation mask by retaining only the selected
        semantic classes.

        Pixels belonging to removed classes are assigned value -1.

        Args:
            mask: numpy array with shape (H, W).

        Returns:
            Filtered mask with the same shape as the input.
            Kept classes are represented by their original class IDs.
            Removed classes are represented by -1.
        """
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                "Mask must be a numpy.ndarray."
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask with 2 dimensions (H, W), "
                f"got shape {mask.shape}."
            )

        filtered_mask = np.full(
            mask.shape,
            -1,
            dtype=np.int16
        )

        for class_id in self.keep_classes:
            filtered_mask[mask == class_id] = class_id

        return filtered_mask

    def get_kept_classes(self):
        """
        Return the original SegFormer class IDs that are retained.

        Returns:
            List of original class IDs.
        """
        return self.keep_classes.copy()

    def get_class_name(self, class_id):
        """
        Return the name of a retained class.

        Args:
            class_id: Original SegFormer class ID.

        Returns:
            Class name.
        """
        if class_id not in self.keep_classes:
            raise ValueError(
                f"Class {class_id} is not part of the reduced classes."
            )

        return self.config.get_class_name(class_id)

    def get_filtered_class_id(self, filtered_index):
        """
        Convert a filtered channel index into the original
        SegFormer class ID.

        Args:
            filtered_index: Index in the filtered probability array.

        Returns:
            Original SegFormer class ID.
        """
        if not 0 <= filtered_index < len(self.filtered_class_ids):
            raise IndexError(
                f"Filtered class index out of range: {filtered_index}"
            )

        return int(self.filtered_class_ids[filtered_index])