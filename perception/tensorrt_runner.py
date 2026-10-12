"""
Minimal TensorRT engine runner: deserializes a serialized engine
(.engine / .plan, e.g. built on the Jetson with trtexec) and runs it
on one input tensor, returning the first output as a numpy array.

Used by SegFormerInference when perception_config.yaml selects
backend: "tensorrt". Kept separate from SegFormerInference so the
ONNX path never imports tensorrt / CUDA bindings, and so the runner
can be reused for any single-input engine.

ENGINE PORTABILITY: a serialized engine only runs with the SAME
TensorRT version and the SAME GPU it was built on. An engine built on
iris (x86, RTX 5090) does not load on the Jetson, and vice versa; an
engine built with another JetPack does not load either. Deserialization
failure raises with this explanation.

NUMPY: trt.nptype() is avoided (see numpy_dtype()): TensorRT 8.5
still uses the np.bool alias, which numpy >= 1.24 removed.

DEVICE WARNING: "[TRT] [W] Using an engine plan file across different
models of devices" means the engine was built on a different device
model (e.g. a Xavier NX dev kit instead of Spot's CORE I/O). It loads
if the GPU architecture matches, but may run slower or misbehave:
rebuild the engine with trtexec on the CORE I/O itself.

PRECISION: an engine built with --fp16 computes in FP16 but keeps FP32
inputs/outputs unless built with --inputIOFormats/--outputIOFormats.
Both cases are handled: the input is cast to the dtype the engine
declares, the output is returned as float32.

TENSORRT API: TensorRT >= 8.5 (JetPack 5.1.x ships 8.5.2) has the
name-based tensor API (set_tensor_address / execute_async_v3); older
versions only the binding-index API (execute_async_v2). The runner
uses the name-based one when available, the binding one otherwise.

CUDA MEMORY: device buffers and the CUDA stream come from the first
of these that works in the virtualenv:
    1. torch with CUDA (NVIDIA's Jetson build) -- the same approach as
       the export/benchmark script that built the engine
       (TRTSegmenter): buffers are torch tensors, the engine runs on
       torch's current CUDA stream;
    2. pycuda;
    3. cuda-python (cuda.cudart).
None of them is a TensorRT dependency.

Buffers are allocated once at construction (static input shape, or
the shape fixed via input_shape for dynamic engines), so run() does
no allocation: host -> device copy, enqueue, device -> host copy,
synchronize.
"""

import atexit

import numpy as np


class TensorRTRunner:

    def __init__(self, engine_path, device_id=0, input_shape=None):
        """
        Parameters
        ----------
        engine_path : str or Path
            Serialized TensorRT engine.

        device_id : int
            CUDA device (0 on the Jetson).

        input_shape : tuple of int, optional
            Required only if the engine has a dynamic input (-1 dims):
            the shape to fix it to, e.g. (1, 3, 512, 512).
        """

        try:
            import tensorrt as trt
        except ImportError as error:
            raise ImportError(
                "The 'tensorrt' Python module is not importable. On the "
                "Jetson it ships with JetPack in the SYSTEM Python "
                "(/usr/lib/python3.8/dist-packages); a virtualenv created "
                "without --system-site-packages does not see it -- symlink "
                "it into the venv's site-packages."
            ) from error

        self.trt = trt
        self.engine_path = str(engine_path)
        self.device_id = int(device_id)

        self.cuda = CudaMemory.create(self.device_id)

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(self.engine_path, "rb") as file:
            engine_bytes = file.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)

        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize the TensorRT engine {self.engine_path}. "
                f"An engine only loads with the TensorRT version and GPU it "
                f"was built with (this is TensorRT {trt.__version__}): "
                "rebuild it on this device with trtexec."
            )

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError("Could not create a TensorRT execution context.")

        self.use_tensor_api = (
            hasattr(self.engine, "num_io_tensors")
            and hasattr(self.context, "execute_async_v3")
        )

        self.tensors = self.describe_tensors()

        inputs = [tensor for tensor in self.tensors if tensor["is_input"]]
        outputs = [tensor for tensor in self.tensors if not tensor["is_input"]]

        if len(inputs) != 1:
            raise ValueError(
                f"Expected an engine with exactly 1 input, got {len(inputs)}: "
                f"{[tensor['name'] for tensor in inputs]}."
            )

        if len(outputs) == 0:
            raise ValueError("The engine has no outputs.")

        self.input = inputs[0]
        self.outputs = outputs

        # SegFormerInference (ONNX path) uses the first ONNX output;
        # TensorRT keeps the ONNX output order.
        self.output = outputs[0]

        self.fix_input_shape(input_shape)
        self.allocate_buffers()

        self.closed = False
        atexit.register(self.close)

    # -----------------------------------------------------------------
    # Engine introspection
    # -----------------------------------------------------------------

    def describe_tensors(self):
        trt = self.trt
        engine = self.engine
        tensors = []

        if self.use_tensor_api:
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)

                tensors.append({
                    "index": index,
                    "name": name,
                    "is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                    "shape": tuple(engine.get_tensor_shape(name)),
                    "dtype": self.numpy_dtype(engine.get_tensor_dtype(name)),
                })
        else:
            for index in range(engine.num_bindings):
                tensors.append({
                    "index": index,
                    "name": engine.get_binding_name(index),
                    "is_input": engine.binding_is_input(index),
                    "shape": tuple(engine.get_binding_shape(index)),
                    "dtype": self.numpy_dtype(engine.get_binding_dtype(index)),
                })

        return tensors

    def numpy_dtype(self, trt_dtype):
        """
        numpy dtype of a TensorRT dtype. trt.nptype() is NOT used: in
        TensorRT 8.5 it references np.bool, removed in numpy >= 1.24,
        and raises AttributeError.
        """

        trt = self.trt

        mapping = {}

        for trt_name, numpy_type in (
            ("float32", np.float32),
            ("float16", np.float16),
            ("int32", np.int32),
            ("int8", np.int8),
            ("uint8", np.uint8),   # TRT >= 8.5
            ("int64", np.int64),   # TRT >= 10
            ("bool", np.bool_),
        ):
            if hasattr(trt, trt_name):
                mapping[getattr(trt, trt_name)] = numpy_type

        if trt_dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")

        return np.dtype(mapping[trt_dtype])

    def fix_input_shape(self, input_shape):
        """Resolve dynamic input dims, then read the actual output shapes."""

        shape = self.input["shape"]

        if any(dim < 0 for dim in shape):
            if input_shape is None:
                raise ValueError(
                    f"Input '{self.input['name']}' is dynamic {shape}: pass "
                    "input_shape to fix it."
                )

            shape = tuple(int(dim) for dim in input_shape)

            if self.use_tensor_api:
                self.context.set_input_shape(self.input["name"], shape)
            else:
                self.context.set_binding_shape(self.input["index"], shape)

        elif input_shape is not None and tuple(input_shape) != shape:
            raise ValueError(
                f"Engine input '{self.input['name']}' is static {shape}, "
                f"requested {tuple(input_shape)}."
            )

        self.input["shape"] = shape

        for output in self.outputs:
            if self.use_tensor_api:
                output["shape"] = tuple(self.context.get_tensor_shape(output["name"]))
            else:
                output["shape"] = tuple(self.context.get_binding_shape(output["index"]))

            if any(dim < 0 for dim in output["shape"]):
                raise ValueError(
                    f"Output '{output['name']}' still has dynamic dims "
                    f"{output['shape']} after fixing the input shape."
                )

    def allocate_buffers(self):
        self.stream = self.cuda.create_stream()

        for tensor in self.tensors:
            nbytes = int(np.prod(tensor["shape"])) * tensor["dtype"].itemsize

            tensor["host"] = np.empty(tensor["shape"], dtype=tensor["dtype"])
            tensor["device"] = self.cuda.malloc(nbytes)
            tensor["nbytes"] = nbytes

            if self.use_tensor_api:
                self.context.set_tensor_address(tensor["name"], int(tensor["device"]))

        # Binding API: device pointers in binding-index order.
        self.bindings = [int(tensor["device"]) for tensor in self.tensors]

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def run(self, input_tensor):
        """
        Run the engine on one input.

        Parameters
        ----------
        input_tensor : numpy.ndarray
            Shape equal to the engine input shape; cast to the engine's
            input dtype if needed.

        Returns
        -------
        numpy.ndarray
            The first output, float32, shape as declared by the engine
            (a copy -- safe to keep across calls).
        """

        if self.closed:
            raise RuntimeError("TensorRTRunner is closed.")

        if tuple(input_tensor.shape) != self.input["shape"]:
            raise ValueError(
                f"Input shape {tuple(input_tensor.shape)} != engine input "
                f"shape {self.input['shape']}."
            )

        host_input = self.input["host"]
        np.copyto(host_input, input_tensor, casting="unsafe")

        self.cuda.memcpy_htod_async(self.input["device"], host_input, self.stream)

        if self.use_tensor_api:
            ok = self.context.execute_async_v3(self.cuda.stream_handle(self.stream))
        else:
            ok = self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.cuda.stream_handle(self.stream)
            )

        if not ok:
            raise RuntimeError("TensorRT execution failed.")

        self.cuda.memcpy_dtoh_async(self.output["host"], self.output["device"], self.stream)
        self.cuda.synchronize(self.stream)

        return self.output["host"].astype(np.float32, copy=True)

    def describe(self):
        """Human-readable summary of the engine I/O."""

        lines = []

        for tensor in self.tensors:
            role = "input " if tensor["is_input"] else "output"
            lines.append(f"{role} {tensor['name']}: {tensor['shape']} {tensor['dtype']}")

        return lines

    def close(self):
        if getattr(self, "closed", True):
            return

        self.closed = True

        for tensor in self.tensors:
            if tensor.get("device") is not None:
                self.cuda.free(tensor["device"])
                tensor["device"] = None

        self.context = None
        self.engine = None
        self.runtime = None

        self.cuda.close()


class CudaMemory:
    """
    Thin adapter over torch, pycuda or cuda-python: just what
    TensorRTRunner needs -- malloc/free, async copies, one stream.
    """

    @staticmethod
    def create(device_id):
        reasons = []

        for adapter in (TorchCudaMemory, PyCudaMemory, CudaPythonMemory):
            try:
                return adapter(device_id)
            except ImportError as error:
                reasons.append(f"{adapter.name}: {error}")

        raise ImportError(
            "TensorRT inference needs torch with CUDA, pycuda or "
            "cuda-python in the virtualenv. Tried:\n  " + "\n  ".join(reasons)
        )


class TorchDevicePointer:
    """A torch uint8 GPU tensor used as a raw device buffer."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __int__(self):
        return int(self.tensor.data_ptr())


class TorchCudaMemory:

    name = "torch"

    def __init__(self, device_id):
        try:
            import torch
        except ImportError as error:
            raise ImportError(f"not installed ({error})") from error

        if not torch.cuda.is_available():
            raise ImportError(
                "installed but CUDA is not available (CPU-only wheel? On "
                "the Jetson NVIDIA's PyTorch build is needed)"
            )

        self.torch = torch
        self.device = torch.device(f"cuda:{device_id}")

        torch.cuda.set_device(self.device)

        # Make sure the CUDA context exists before TensorRT uses it.
        torch.cuda.init()

    def malloc(self, nbytes):
        tensor = self.torch.empty(
            max(int(nbytes), 1), dtype=self.torch.uint8, device=self.device
        )

        return TorchDevicePointer(tensor)

    @staticmethod
    def free(pointer):
        pointer.tensor = None

    def create_stream(self):
        # torch's current stream: copies done with torch below are
        # queued on the same stream as the engine.
        return self.torch.cuda.current_stream(self.device)

    @staticmethod
    def stream_handle(stream):
        return stream.cuda_stream

    def memcpy_htod_async(self, device, host, stream):
        source = self.torch.from_numpy(host.reshape(-1).view(np.uint8))

        with self.torch.cuda.stream(stream):
            device.tensor[:source.numel()].copy_(source, non_blocking=False)

    def memcpy_dtoh_async(self, host, device, stream):
        with self.torch.cuda.stream(stream):
            data = device.tensor[:host.nbytes].cpu().numpy()

        host.reshape(-1).view(np.uint8)[:] = data

    @staticmethod
    def synchronize(stream):
        stream.synchronize()

    def close(self):
        pass


class PyCudaMemory:

    name = "pycuda"

    def __init__(self, device_id):
        import pycuda.driver as cuda

        self.driver = cuda

        cuda.init()

        # Own context, made current on this (the main) thread. All
        # inference happens on the thread that created the runner.
        self.context = cuda.Device(device_id).make_context()

    def malloc(self, nbytes):
        return self.driver.mem_alloc(max(int(nbytes), 1))

    def free(self, pointer):
        pointer.free()

    def create_stream(self):
        return self.driver.Stream()

    @staticmethod
    def stream_handle(stream):
        return stream.handle

    def memcpy_htod_async(self, device, host, stream):
        self.driver.memcpy_htod_async(device, host, stream)

    def memcpy_dtoh_async(self, host, device, stream):
        self.driver.memcpy_dtoh_async(host, device, stream)

    @staticmethod
    def synchronize(stream):
        stream.synchronize()

    def close(self):
        if self.context is not None:
            try:
                self.context.pop()
            except Exception:
                pass

            self.context = None


class CudaPythonMemory:

    name = "cuda-python"

    def __init__(self, device_id):
        from cuda import cudart

        self.cudart = cudart
        self.check(cudart.cudaSetDevice(device_id), "cudaSetDevice")
        self.streams = []

    def check(self, result, call):
        error = result[0]

        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{call} failed: {error}")

        return result[1] if len(result) > 1 else None

    def malloc(self, nbytes):
        return self.check(self.cudart.cudaMalloc(max(int(nbytes), 1)), "cudaMalloc")

    def free(self, pointer):
        self.cudart.cudaFree(pointer)

    def create_stream(self):
        stream = self.check(self.cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.streams.append(stream)

        return stream

    @staticmethod
    def stream_handle(stream):
        return int(stream)

    def memcpy_htod_async(self, device, host, stream):
        self.check(self.cudart.cudaMemcpyAsync(
            device, host.ctypes.data, host.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
        ), "cudaMemcpyAsync H2D")

    def memcpy_dtoh_async(self, host, device, stream):
        self.check(self.cudart.cudaMemcpyAsync(
            host.ctypes.data, device, host.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
        ), "cudaMemcpyAsync D2H")

    def synchronize(self, stream):
        self.check(self.cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def close(self):
        for stream in self.streams:
            self.cudart.cudaStreamDestroy(stream)

        self.streams = []