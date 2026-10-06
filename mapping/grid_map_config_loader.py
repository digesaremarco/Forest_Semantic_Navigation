from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GridMapConfig:
    """Loads and provides access to the grid map configuration."""

    # Used only if the 'noise' section is missing entirely:
    # alpha = 0 disables the distance weighting (plain mean).
    DEFAULT_NOISE = {
        "sigma0": 0.01,
        "alpha": 0.0,
        "exponent": 4.0,
        "reference_distance": 1.0,
    }

    HEIGHT_DATUM_MODES = ("none", "ground_first_frame")

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

        self.load_map_config()
        self.load_bounds_config()
        self.load_noise_config()
        self.load_height_datum_config()

    def load_map_config(self):
        map_config = self._config.get("map")

        if map_config is None:
            raise ValueError(
                "Missing 'map' section in grid map configuration."
            )

        self.map_length = map_config.get("map_length")
        self.resolution = map_config.get("resolution")

        if self.map_length is None:
            raise ValueError(
                "Missing 'map.map_length' in grid map configuration."
            )

        if self.resolution is None:
            raise ValueError(
                "Missing 'map.resolution' in grid map configuration."
            )

        self.validate_positive(self.map_length, "map.map_length")
        self.validate_positive(self.resolution, "map.resolution")

        if self.resolution > self.map_length:
            raise ValueError(
                "map.resolution must not be larger than map.map_length."
            )

        self.cell_n = self.compute_cell_n(
            self.map_length,
            self.resolution
        )

    def load_bounds_config(self):
        self.out_of_bounds_warning = self._config.get(
            "out_of_bounds_warning", True
        )

    def load_noise_config(self):
        """
        Load the depth noise model used by GridMap to weight points:

            sigma^2(z) = sigma0^2 + alpha * z^exponent

        Exposed as noise_sigma0, noise_alpha, noise_exponent and
        noise_reference_distance, the attribute names GridMap reads.

        The 'noise' section is optional for backward compatibility:
        if it is missing, defaults with alpha = 0 are used (no
        weighting) and a warning is printed, so the weighting is
        never disabled silently. If the section is present, every
        key in it is required.
        """

        noise_config = self._config.get("noise")

        if noise_config is None:
            print(
                f"[GridMapConfig] WARNING: no 'noise' section in "
                f"{self.config_path}; distance weighting DISABLED "
                f"(defaults: {self.DEFAULT_NOISE})."
            )
            noise_config = self.DEFAULT_NOISE

        values = {}

        for key in ("sigma0", "alpha", "exponent", "reference_distance"):
            if key not in noise_config:
                raise ValueError(
                    f"Missing 'noise.{key}' in grid map configuration."
                )

            values[key] = self.to_float(noise_config[key], f"noise.{key}")

        self.validate_positive(values["sigma0"], "noise.sigma0")
        self.validate_non_negative(values["alpha"], "noise.alpha")
        self.validate_non_negative(values["exponent"], "noise.exponent")
        self.validate_positive(
            values["reference_distance"], "noise.reference_distance"
        )

        self.noise_sigma0 = values["sigma0"]
        self.noise_alpha = values["alpha"]
        self.noise_exponent = values["exponent"]
        self.noise_reference_distance = values["reference_distance"]

    def load_height_datum_config(self):
        """
        Load the vertical datum settings (see grid_map_config.yaml).

        Exposed as height_datum_mode ("none" | "ground_first_frame")
        and height_datum_<param> attributes, the names GridMap reads.

        The section is optional for backward compatibility: if it is
        missing, mode "none" (raw odom heights) is used and a warning
        is printed. If present with mode ground_first_frame, every
        parameter is required.
        """

        section = self._config.get("height_datum")

        if section is None:
            print(
                f"[GridMapConfig] WARNING: no 'height_datum' section in "
                f"{self.config_path}; elevations stay in raw odom "
                "heights (mode: none)."
            )
            section = {"mode": "none"}

        mode = section.get("mode")

        if mode not in self.HEIGHT_DATUM_MODES:
            raise ValueError(
                f"height_datum.mode must be one of "
                f"{self.HEIGHT_DATUM_MODES}, got {mode!r}."
            )

        self.height_datum_mode = mode

        if mode == "none":
            return

        float_keys = (
            "min_range", "max_range", "ransac_threshold", "max_tilt_deg"
        )
        int_keys = ("ransac_iterations", "min_inliers")

        for key in float_keys + int_keys:
            if key not in section:
                raise ValueError(
                    f"Missing 'height_datum.{key}' in grid map "
                    "configuration."
                )

        for key in float_keys:
            value = self.to_float(section[key], f"height_datum.{key}")
            self.validate_positive(value, f"height_datum.{key}")
            setattr(self, f"height_datum_{key}", value)

        for key in int_keys:
            value = section[key]

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"height_datum.{key} must be an integer, got {value!r}."
                )

            self.validate_positive(value, f"height_datum.{key}")
            setattr(self, f"height_datum_{key}", value)

        if self.height_datum_min_range >= self.height_datum_max_range:
            raise ValueError(
                "height_datum.min_range must be smaller than "
                "height_datum.max_range."
            )

        if self.height_datum_max_tilt_deg >= 90:
            raise ValueError("height_datum.max_tilt_deg must be < 90.")

    @staticmethod
    def compute_cell_n(map_length, resolution):
        """
        Compute the number of cells per side of the square grid.

        Kept odd on purpose, so the grid has an exact center cell:
        that center cell is where the first pose received by
        GridMap.update() gets anchored.
        """

        cell_n = int(round(map_length / resolution))

        if cell_n % 2 == 0:
            cell_n += 1

        return cell_n

    @staticmethod
    def to_float(value, parameter_name):
        """
        Convert a YAML value to float.

        PyYAML (YAML 1.1) parses scientific notation without a
        decimal point, e.g. '1e-5', as a STRING, not a float; this
        accepts such strings instead of failing later in arithmetic.
        Booleans are rejected explicitly (bool is a subclass of int).
        """

        if isinstance(value, bool):
            raise ValueError(f"{parameter_name} must be a numeric value.")

        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{parameter_name} must be a numeric value, got {value!r}."
            ) from None

    @staticmethod
    def validate_positive(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{parameter_name} must be a numeric value."
            )

        if value <= 0:
            raise ValueError(
                f"{parameter_name} must be greater than zero."
            )

    @staticmethod
    def validate_non_negative(value, parameter_name):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{parameter_name} must be a numeric value."
            )

        if value < 0:
            raise ValueError(
                f"{parameter_name} must be greater than or equal to zero."
            )