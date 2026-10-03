"""
SegFormer ONNX inference for forest semantic segmentation.

This module loads the exported SegFormer ONNX model and provides
methods for preprocessing RGB images and running semantic
segmentation inference using ONNX Runtime.

If an NVIDIA GPU (CUDAExecutionProvider) is not available,
inference automatically falls back to the CPU.
"""

import time

import cv2
import numpy as np

import onnxruntime as ort

# preload_dlls() exists only in recent onnxruntime(-gpu) builds;
# it may fail or be missing on CPU-only installations.
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception as e:
        print(f"Warning: ort.preload_dlls() failed: {e}")


class SegFormerInference:
    """ONNX inference wrapper for the SegFormer model."""

    def __init__(self, config):
        self.config = config
        requested_device = str(config.device).lower()

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

        # Parse requested device: "cpu", "cuda0", "cuda1", ...
        self.device_id = None

        if requested_device == "cpu":
            want_cuda = False
        elif requested_device.startswith("cuda"):
            want_cuda = True
            suffix = requested_device.replace("cuda", "") or "0"
            try:
                self.device_id = int(suffix)
            except ValueError:
                raise ValueError(
                    f"Invalid device: {requested_device}. "
                    "Expected 'cpu', 'cuda0', 'cuda1', etc."
                )
        else:
            raise ValueError(
                f"Invalid device: {requested_device}. "
                "Expected 'cpu', 'cuda0', 'cuda1', etc."
            )

        # Check ONNX Runtime
        available_providers = ort.get_available_providers()

        print("\nONNX Runtime")
        print(f"Version            : {ort.__version__}")
        print(f"Available providers: {available_providers}")

        self.session = None

        # Try CUDA first (if requested and available)
        if want_cuda:
            if "CUDAExecutionProvider" in available_providers:
                try:
                    self.session = ort.InferenceSession(
                        str(self.model_path),
                        providers=[
                            (
                                "CUDAExecutionProvider",
                                {"device_id": self.device_id}
                            ),
                            "CPUExecutionProvider"
                        ]
                    )

                    # ONNX Runtime may silently fall back to CPU
                    if ("CUDAExecutionProvider"
                            not in self.session.get_providers()):
                        print(
                            "Warning: CUDA session could not be "
                            "activated, falling back to CPU."
                        )
                        self.session = None

                except Exception as e:
                    print(
                        f"Warning: failed to create CUDA session "
                        f"({e}). Falling back to CPU."
                    )
                    self.session = None
            else:
                print(
                    "Warning: CUDAExecutionProvider not available. "
                    "Falling back to CPU."
                )

        # CPU fallback (or CPU explicitly requested)
        if self.session is None:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )
            self.device = "cpu"
            self.device_id = None
        else:
            self.device = f"cuda{self.device_id}"

        session_providers = self.session.get_providers()

        print(f"Selected device    : {self.device}")
        print(f"Session providers  : {session_providers}")

        # Model input/output information
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        """
        Preprocess an RGB image before ONNX inference.

        Pipeline:
            RGB image -> Resize 512x512 -> float32 -> [0,1]
            -> ImageNet normalization -> HWC->CHW -> batch dim

        Args:
            image (np.ndarray): RGB image with shape (H, W, 3).

        Returns:
            np.ndarray: Tensor with shape (1, 3, 512, 512).
        """

        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected RGB image with shape "
                f"(H, W, 3), got {image.shape}."
            )

        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LINEAR
        )

        image = image.astype(np.float32)
        image /= 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image, dtype=np.float32)

        return image

    def infer(self, image):
        """
        Run semantic segmentation inference (GPU or CPU).

        Args:
            image (np.ndarray): RGB input image with shape (H, W, 3).

        Returns:
            np.ndarray: Model output with shape (C, 512, 512).
        """

        input_tensor = self.preprocess(image)

        start_time = time.perf_counter()

        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        inference_time = time.perf_counter() - start_time

        # Remove batch dimension
        probabilities = output[0]

        print(
            f"{self.device.upper()} inference time: "
            f"{inference_time * 1000:.2f} ms"
        )

        return probabilities

    def get_segmentation_mask(self, probabilities):
        """
        Convert model output into a segmentation mask.

        Args:
            probabilities (np.ndarray): Model output with shape (C, H, W).

        Returns:
            np.ndarray: Segmentation mask with shape (H, W).
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

        mask = np.argmax(probabilities, axis=0).astype(np.uint8)

        return mask