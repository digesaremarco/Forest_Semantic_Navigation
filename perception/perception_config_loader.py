from pathlib import Path
import yaml


class PerceptionConfig:
    """Loads and provides access to the perception configuration"""

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

        self.model_path = model_config.get("path")
        self.num_classes = model_config.get("num_classes")

        if self.model_path is None:
            raise ValueError(
                "Missing 'model.path' in perception configuration."
            )

        if self.num_classes is None:
            raise ValueError(
                "Missing 'model.num_classes' in perception configuration."
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
        """Return the name associated with a class ID"""
        return self.classes[class_id]["name"]

    def get_class_rgb(self, class_id):
        """Return the RGB color associated with a class ID"""
        return self.classes[class_id]["rgb"]

    def get_class(self, class_id):
        """Return the complete information of a class"""
        return self.classes[class_id]