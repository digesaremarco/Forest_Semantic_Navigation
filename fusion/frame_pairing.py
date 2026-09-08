"""
Offline RGB-D frame pairing utility.

Finds the closest depth frame for each RGB image using the
timestamps contained in the Spot filenames and creates a
1:1 correspondence by renaming the paired files.

This script is intended for offline testing only.
It is not part of the live Spot pipeline.
"""

import re
import shutil
from datetime import datetime
from pathlib import Path


class FramePairer:
    """Finds and creates 1:1 RGB-depth frame correspondences."""

    def __init__(
        self,
        rgb_dir,
        depth_dir,
        output_rgb_dir,
        output_depth_dir,
        max_time_difference=0.1
    ):
        """
        Initialize the frame pairer.

        Args:
            rgb_dir: Directory containing RGB images.
            depth_dir: Directory containing depth images.
            output_rgb_dir: Directory for paired RGB images.
            output_depth_dir: Directory for paired depth images.
            max_time_difference: Maximum allowed temporal difference
                between RGB and depth frames, in seconds.
        """
        self.rgb_dir = Path(rgb_dir)
        self.depth_dir = Path(depth_dir)

        self.output_rgb_dir = Path(output_rgb_dir)
        self.output_depth_dir = Path(output_depth_dir)

        self.max_time_difference = max_time_difference

        self.rgb_data = {}
        self.depth_data = {}

        self.matched_pairs = []

    def extract_timestamp(self, filename):
        """
        Extract timestamp from a Spot filename.

        Expected format:

            AAAAMMGG_HHMMSS_mmm

        Returns:
            Timestamp in seconds, or None if no timestamp is found.
        """
        match = re.search(
            r"(\d{8})_(\d{6})_(\d{3})",
            filename
        )

        if match is None:
            return None

        date_str = match.group(1)
        time_str = match.group(2)
        ms_str = match.group(3)

        dt = datetime.strptime(
            f"{date_str}_{time_str}",
            "%Y%m%d_%H%M%S"
        )

        return dt.timestamp() + (
            float(ms_str) / 1000.0
        )

    def find_files(self):
        """Find RGB and depth files and extract their timestamps."""

        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}"
            )

        if not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}"
            )

        rgb_files = sorted(
            self.rgb_dir.glob("*.jpg")
        )

        depth_files = sorted(
            self.depth_dir.glob("*.png")
        )

        for file_path in rgb_files:
            timestamp = self.extract_timestamp(
                file_path.name
            )

            if timestamp is not None:
                self.rgb_data[file_path] = timestamp

        for file_path in depth_files:
            timestamp = self.extract_timestamp(
                file_path.name
            )

            if timestamp is not None:
                self.depth_data[file_path] = timestamp

        print(f"RGB files found   : {len(rgb_files)}")
        print(f"RGB timestamps    : {len(self.rgb_data)}")
        print(f"Depth files found : {len(depth_files)}")
        print(f"Depth timestamps  : {len(self.depth_data)}")

    def find_pairs(self):
        """
        Find the closest depth frame for every RGB frame.

        Returns:
            List of tuples:
                (rgb_path, depth_path, time_difference)
        """
        self.matched_pairs = []

        for rgb_path, rgb_time in self.rgb_data.items():

            best_match = None
            min_diff = float("inf")

            for depth_path, depth_time in self.depth_data.items():

                diff = abs(rgb_time - depth_time)

                if diff < min_diff:
                    min_diff = diff
                    best_match = depth_path

            if (
                best_match is not None
                and min_diff <= self.max_time_difference
            ):
                self.matched_pairs.append(
                    (
                        rgb_path,
                        best_match,
                        min_diff
                    )
                )

        return self.matched_pairs

    def create_output_directories(self):
        """Create output directories."""

        self.output_rgb_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_depth_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def create_paired_dataset(self):
        """
        Copy matched RGB-depth pairs into the output directories
        using a common sequential filename.
        """

        self.create_output_directories()

        for index, (
            rgb_path,
            depth_path,
            time_difference
        ) in enumerate(self.matched_pairs):

            filename = f"{index:06d}"

            output_rgb = (
                self.output_rgb_dir /
                f"{filename}{rgb_path.suffix.lower()}"
            )

            output_depth = (
                self.output_depth_dir /
                f"{filename}{depth_path.suffix.lower()}"
            )

            shutil.copy2(
                rgb_path,
                output_rgb
            )

            shutil.copy2(
                depth_path,
                output_depth
            )

    def print_results(self, num_examples=5):
        """Print pairing results."""

        print("\n" + "-" * 60)
        print("Pairing results")
        print("-" * 60)

        for index, (
            rgb_path,
            depth_path,
            time_difference
        ) in enumerate(
            self.matched_pairs[:num_examples]
        ):

            print(f"Pair {index:06d}")
            print(f"  RGB   : {rgb_path.name}")
            print(f"  Depth : {depth_path.name}")
            print(
                f"  Difference : "
                f"{time_difference * 1000:.1f} ms"
            )
            print()

    def run(self):
        """Run the complete pairing and copying process."""

        self.find_files()
        self.find_pairs()

        print(
            f"\nMatched pairs : "
            f"{len(self.matched_pairs)}"
        )

        self.print_results()

        self.create_paired_dataset()

        print("-" * 60)
        print("Paired dataset created successfully.")
        print(f"RGB output   : {self.output_rgb_dir}")
        print(f"Depth output : {self.output_depth_dir}")
        print("-" * 60)


if __name__ == "__main__":

    print("=" * 60)
    print("Offline RGB-D Frame Pairing")
    print("=" * 60)

    # Test dataset
    project_root = Path(__file__).resolve().parent.parent

    testset_dir = project_root / "testset"

    rgb_dir = testset_dir / "images"
    depth_dir = testset_dir / "depths"

    # Output directories containing the paired dataset.
    output_rgb_dir = testset_dir / "paired_images"
    output_depth_dir = testset_dir / "paired_depths"

    # Maximum allowed RGB-depth difference.
    # 100 ms = 0.1 seconds.
    max_time_difference = 0.1

    # Initialize
    pairer = FramePairer(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        output_rgb_dir=output_rgb_dir,
        output_depth_dir=output_depth_dir,
        max_time_difference=max_time_difference
    )

    # Run
    pairer.run()

    print("\n" + "=" * 60)
    print("Test completed.")
    print("=" * 60)