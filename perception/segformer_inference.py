"""
SegFormer inference for forest semantic segmentation, with two
interchangeable backends selected by perception_config.yaml
(model.backend):

    "onnx"      ONNX Runtime on the exported ONNX model (model.path).
                CUDAExecutionProvider if requested and available,
                otherwise automatic fallback to the CPU. Used on iris /
                any x86 machine.

    "tensorrt"  Serialized TensorRT engine (model.engine_path), via
                perception/tensorrt_runner.py. GPU only, no fallback:
                if the engine cannot be loaded, construction fails
                rather than silently running at CPU speed. Used on the
                Jetson (CORE I/O).

Both backends get exactly the same preprocessing (preprocess()) and
return the same thing from infer(): the model output without the
batch dimension, (C, H_out, W_out) float32. Everything downstream
(ClassReducer, PointCloudBuilder) is backend-agnostic.

Each backend's Python dependencies are imported only when that
backend is selected: the ONNX path never needs tensorrt/pycuda, the
TensorRT path never needs onnxruntime.

The engine must come from the SAME ONNX model as model.path (the
Tversky checkpoint) -- this module cannot check that; compare the two
backends on the same image with test/test_segformer_backends.py.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SUPPORTED_BACKENDS = ("onnx", "tensorrt")


class SegFormerInference:
    """ONNX Runtime / TensorRT inference wrapper for the SegFormer model."""

    def __init__(self, config, backend=None, verbose=True):
        """
        Parameters
        ----------
        config : PerceptionConfig

        backend : str, optional
            Override of config.backend ("onnx" / "tensorrt"), e.g. to
            compare the two on the same machine.

        verbose : bool
            Print the per-call inference time.
        """

        self.config = config
        self.verbose = verbose

        # Model configuration
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

        self.backend = str(backend if backend is not None else config.backend).lower()

        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Invalid backend: {self.backend!r}. Expected one of "
                f"{SUPPORTED_BACKENDS}."
            )

        want_cuda, self.device_id = self.parse_device(config.device)

        # Set by the backend initializer.
        self.session = None
        self.trt_runner = None

        if self.backend == "onnx":
            self.model_path = self.resolve_path(config.model_path)
            self.init_onnx(want_cuda)
        else:
            if not want_cuda:
                raise ValueError(
                    "backend 'tensorrt' needs a CUDA device "
                    f"(model.device = {config.device!r})."
                )

            self.model_path = self.resolve_path(config.engine_path)
            self.init_tensorrt()

        # Warm-up: the first call (CUDA kernels, TensorRT tactics,
        # memory pools) is much slower than steady state.
        self.run_backend(
            np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.float32)
        )

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    @staticmethod
    def parse_device(device):
        """'cpu' -> (False, None); 'cuda0', 'cuda1', ... -> (True, id)."""

        requested_device = str(device).lower()

        if requested_device == "cpu":
            return False, None

        if requested_device.startswith("cuda"):
            suffix = requested_device.replace("cuda", "") or "0"

            try:
                return True, int(suffix)
            except ValueError:
                pass

        raise ValueError(
            f"Invalid device: {requested_device}. "
            "Expected 'cpu', 'cuda0', 'cuda1', etc."
        )

    @staticmethod
    def resolve_path(path):
        """Relative paths are relative to the project root."""

        if path is None:
            raise ValueError("Model path not set in perception_config.yaml.")

        path = Path(path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        return path

    def init_onnx(self, want_cuda):
        import onnxruntime as ort

        # preload_dlls() exists only in recent onnxruntime(-gpu) builds;
        # it may fail or be missing on CPU-only installations.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:
                print(f"Warning: ort.preload_dlls() failed: {e}")

        available_providers = ort.get_available_providers()

        print("\nONNX Runtime")
        print(f"Version            : {ort.__version__}")
        print(f"Model              : {self.model_path}")
        print(f"Available providers: {available_providers}")

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

        print(f"Selected device    : {self.device}")
        print(f"Session providers  : {self.session.get_providers()}")

        # Model input/output information
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def init_tensorrt(self):
        from perception.tensorrt_runner import TensorRTRunner

        self.trt_runner = TensorRTRunner(
            self.model_path,
            device_id=self.device_id,
            input_shape=(1, 3, self.image_size[1], self.image_size[0])
        )

        self.device = f"cuda{self.device_id}"

        output_shape = self.trt_runner.output["shape"]

        if len(output_shape) != 4 or output_shape[1] != self.num_classes:
            raise ValueError(
                f"Engine output shape {output_shape}: expected "
                f"(1, {self.num_classes}, H, W) -- is this the SegFormer "
                f"engine with {self.num_classes} classes?"
            )

        print("\nTensorRT")
        print(f"Version            : {self.trt_runner.trt.__version__}")
        print(f"Engine             : {self.model_path}")
        print(f"CUDA bindings      : {self.trt_runner.cuda.name}")
        print(f"API                : {'tensor (v3)' if self.trt_runner.use_tensor_api else 'binding (v2)'}")
        print(f"Selected device    : {self.device}")

        for line in self.trt_runner.describe():
            print(f"  {line}")

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def preprocess(self, image):
        """
        Preprocess an RGB image before inference.

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

    def run_backend(self, input_tensor):
        """Model output WITH batch dimension, float32."""

        if self.backend == "onnx":
            return self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )[0]

        return self.trt_runner.run(input_tensor)

    def infer(self, image):
        """
        Run semantic segmentation inference.

        Args:
            image (np.ndarray): RGB input image with shape (H, W, 3).

        Returns:
            np.ndarray: Model output with shape (C, H_out, W_out),
            float32.
        """

        input_tensor = self.preprocess(image)

        start_time = time.perf_counter()

        output = self.run_backend(input_tensor)

        inference_time = time.perf_counter() - start_time

        # Remove batch dimension
        probabilities = output[0]

        if self.verbose:
            print(
                f"{self.backend.upper()} {self.device.upper()} inference time: "
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

    def close(self):
        """Release the TensorRT engine and GPU buffers (no-op for ONNX)."""

        if self.trt_runner is not None:
            self.trt_runner.close()
            self.trt_runner = None