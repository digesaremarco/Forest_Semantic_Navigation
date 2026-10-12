from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_BACKENDS = ("onnx", "tensorrt")


class PerceptionConfig:
    """Loads and provides access to the perception configuration."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

        if self._config is None:
            raise ValueError(
                f"Configuration file is empty: {self.config_path}"
            )

        self.load_model_config()
        self.load_classes()

    def load_model_config(self):
        model_config = self._config.get("model")

        if model_config is None:
            raise ValueError(
                "Missing 'model' section in perception configuration."
            )

        # Inference backend. Missing -> "onnx", the behavior before
        # the field existed.
        backend = model_config.get("backend", "onnx")

        if not isinstance(backend, str) or backend.lower() not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"'model.backend' must be one of {SUPPORTED_BACKENDS}, "
                f"got {backend!r}."
            )

        self.backend = backend.lower()

        self.num_classes = model_config.get("num_classes")

        if self.num_classes is None:
            raise ValueError(
                "Missing 'model.num_classes' in perception configuration."
            )

        self.device = model_config.get("device")

        if self.device is None:
            raise ValueError(
                "Missing 'model.device' in perception configuration."
            )

        if self.backend == "tensorrt" and not str(self.device).lower().startswith("cuda"):
            raise ValueError(
                "'model.backend' tensorrt needs a CUDA device, "
                f"got model.device = {self.device!r}."
            )

        # Both paths are resolved when present; only the one used by
        # the selected backend is required to exist (the Jetson may
        # not have the ONNX, iris cannot use the engine).
        self.model_path = self.resolve_path(model_config.get("path"))
        self.engine_path = self.resolve_path(model_config.get("engine_path"))

        if self.backend == "onnx":
            self.require_file(self.model_path, "model.path", "ONNX model")
        else:
            self.require_file(self.engine_path, "model.engine_path", "TensorRT engine")

    @staticmethod
    def resolve_path(path):
        """Absolute Path (relative to the project root), or None."""

        if path is None:
            return None

        path = Path(path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def require_file(self, path, parameter_name, description):
        if path is None:
            raise ValueError(
                f"Missing '{parameter_name}' in perception configuration "
                f"(required by backend '{self.backend}')."
            )

        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    def load_classes(self):
        classes_config = self._config.get("classes")

        if classes_config is None:
            raise ValueError(
                "Missing 'classes' section in perception configuration."
            )

        self.classes = {}

        for class_id, class_data in classes_config.items():
            class_id = int(class_id)

            name = class_data.get("name")
            rgb = class_data.get("rgb")

            if name is None:
                raise ValueError(
                    f"Missing name for class {class_id}."
                )

            if rgb is None:
                raise ValueError(
                    f"Missing RGB value for class {class_id}."
                )

            self.classes[class_id] = {
                "name": name,
                "rgb": tuple(rgb)
            }

    def get_class_name(self, class_id):
        """Return the name associated with a class ID."""
        return self.classes[class_id]["name"]

    def get_class_rgb(self, class_id):
        """Return the RGB color associated with a class ID."""
        return self.classes[class_id]["rgb"]

    def get_class(self, class_id):
        """Return the complete information of a class."""
        return self.classes[class_id]