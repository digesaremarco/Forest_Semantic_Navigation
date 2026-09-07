"""
SegFormer ONNX inference for forest semantic segmentation.

This module loads the exported SegFormer ONNX model and provides
methods for preprocessing RGB images and running semantic
segmentation inference on an NVIDIA GPU using ONNX Runtime.
"""

import time

import cv2
import numpy as np

import onnxruntime as ort
ort.preload_dlls()


class SegFormerInference:
    """ONNX inference wrapper for the SegFormer model."""

    def __init__(self, config):
        self.config = config
        self.device = config.device.lower()

        # Convert "cuda0"/"cuda1" -> 0/1
        if not self.device.startswith("cuda"):
            raise ValueError(
                f"Invalid CUDA device: {self.device}. "
                "Expected 'cuda0', 'cuda1', etc."
            )

        try:
            self.device_id = int(
                self.device.replace("cuda", "")
            )
        except ValueError:
            raise ValueError(
                f"Invalid CUDA device: {self.device}. "
                "Expected 'cuda0', 'cuda1', etc."
            )

        # Model configuration
        self.model_path = config.model_path
        self.num_classes = config.num_classes

        # Input preprocessing parameters
        self.image_size = (512, 512)

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        # Check ONNX Runtime
        available_providers = ort.get_available_providers()

        print("\nONNX Runtime")
        print(f"Version            : {ort.__version__}")
        print(
            f"Available providers: "
            f"{available_providers}"
        )

        # CUDA MUST be available
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is not available.\n"
                f"Available providers: {available_providers}\n\n"
                "Make sure onnxruntime-gpu and the required "
                "CUDA/cuDNN libraries are installed correctly."
            )

        # ----------------------------------------------------
        # Check requested CUDA device
        # ----------------------------------------------------

        print(
            f"Selected CUDA device: "
            f"{self.device}"
        )

        print(
            f"CUDA device ID      : "
            f"{self.device_id}"
        )

        # ----------------------------------------------------
        # Create CUDA session
        # ----------------------------------------------------

        cuda_options = {
            "device_id": self.device_id
        }

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[
                (
                    "CUDAExecutionProvider",
                    cuda_options
                )
            ]
        )

        # Verify session providers
        session_providers = self.session.get_providers()

        print(
            f"Session providers  : "
            f"{session_providers}"
        )

        if "CUDAExecutionProvider" not in session_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is not active "
                "in the ONNX Runtime session.\n"
                f"Session providers: {session_providers}"
            )

        # Model input/output information
        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.output_name = (
            self.session.get_outputs()[0].name
        )

    def preprocess(self, image):
        """
        Preprocess an RGB image before ONNX inference.

        Pipeline:

            RGB image
                ↓
            Resize 512x512
                ↓
            Convert to float32
                ↓
            Scale [0,255] -> [0,1]
                ↓
            ImageNet normalization
                ↓
            HWC -> CHW
                ↓
            Add batch dimension

        Args:
            image (np.ndarray):
                RGB image with shape (H, W, 3).

        Returns:
            np.ndarray:
                Tensor with shape (1, 3, 512, 512).
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a numpy array."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected RGB image with shape "
                f"(H, W, 3), got {image.shape}."
            )

        # Resize
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to float32
        image = image.astype(
            np.float32
        )

        # Scale [0,255] -> [0,1]
        image /= 255.0

        # ImageNet normalization
        image = (
            image - self.mean
        ) / self.std

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # Ensure contiguous float32 memory
        image = np.ascontiguousarray(
            image,
            dtype=np.float32
        )

        return image

    def infer(self, image):
        """
        Run semantic segmentation inference on the GPU.

        Args:
            image (np.ndarray):
                RGB input image with shape (H, W, 3).

        Returns:
            np.ndarray:
                Model output with shape (24, 512, 512).

        """

        # Preprocess
        input_tensor = self.preprocess(
            image
        )

        # GPU inference
        start_time = time.perf_counter()

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )[0]

        inference_time = (
            time.perf_counter()
            - start_time
        )

        # Remove batch dimension
        probabilities = output[0]

        print(
            f"GPU {self.device} inference time: "
            f"{inference_time * 1000:.2f} ms"
        )

        return probabilities

    def get_segmentation_mask(
        self,
        probabilities
    ):
        """
        Convert model output into a segmentation mask.

        Args:
            probabilities (np.ndarray):
                Model output with shape (C, H, W).

        Returns:
            np.ndarray:
                Segmentation mask with shape (H, W).
        """

        if probabilities.ndim != 3:
            raise ValueError(
                "Expected probabilities with shape "
                f"(C, H, W), got {probabilities.shape}."
            )

        if probabilities.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, "
                f"got {probabilities.shape[0]}."
            )

        mask = np.argmax(
            probabilities,
            axis=0
        ).astype(np.uint8)

        return mask