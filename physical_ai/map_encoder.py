from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def read_open3d_binary_ply(path: str | Path) -> np.ndarray:
    """Memory-map the six-double Open3D PLY layout supplied by the contest."""
    path = Path(path)
    with path.open("rb") as handle:
        header = b""
        vertex_count = None
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            header += line
            if line.startswith(b"element vertex "):
                vertex_count = int(line.split()[-1])
    if vertex_count is None:
        raise ValueError("PLY vertex count is missing")
    return np.memmap(path, dtype="<f8", mode="r", offset=len(header), shape=(vertex_count, 6))


@dataclass
class MapRaster:
    minimum_xy: np.ndarray
    resolution: float
    height: np.ndarray
    log_density: np.ndarray

    @classmethod
    def from_point_cloud(
        cls,
        path: str | Path,
        minimum_xy: np.ndarray,
        maximum_xy: np.ndarray,
        resolution: float = 2.0,
    ) -> "MapRaster":
        points = read_open3d_binary_ply(path)
        minimum_xy = np.asarray(minimum_xy, dtype=np.float64)
        maximum_xy = np.asarray(maximum_xy, dtype=np.float64)
        shape_xy = np.ceil((maximum_xy - minimum_xy) / resolution).astype(int) + 1
        height = np.zeros((shape_xy[1], shape_xy[0]), dtype=np.float32)
        count = np.zeros_like(height, dtype=np.int32)
        chunk = 250_000
        for start in range(0, len(points), chunk):
            xyz = np.asarray(points[start : start + chunk, :3])
            cell = np.floor((xyz[:, :2] - minimum_xy) / resolution).astype(np.int64)
            valid = (
                (cell[:, 0] >= 0)
                & (cell[:, 0] < shape_xy[0])
                & (cell[:, 1] >= 0)
                & (cell[:, 1] < shape_xy[1])
            )
            x, y, z = cell[valid, 0], cell[valid, 1], xyz[valid, 2].astype(np.float32)
            np.maximum.at(height, (y, x), z)
            np.add.at(count, (y, x), 1)
        return cls(minimum_xy, float(resolution), height, np.log1p(count).astype(np.float32))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            minimum_xy=self.minimum_xy,
            resolution=np.array(self.resolution),
            height=self.height,
            log_density=self.log_density,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MapRaster":
        raw = np.load(path)
        return cls(
            minimum_xy=raw["minimum_xy"],
            resolution=float(raw["resolution"]),
            height=raw["height"],
            log_density=raw["log_density"],
        )

    def sample(self, image: np.ndarray, xy: np.ndarray) -> np.ndarray:
        cell = (np.asarray(xy) - self.minimum_xy) / self.resolution
        coordinates = np.stack((cell[..., 1], cell[..., 0]), axis=0)
        return map_coordinates(image, coordinates, order=1, mode="nearest")

    def context_features(
        self,
        positions: np.ndarray,
        bs_position: np.ndarray,
        corridor_samples: int = 32,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        bs_position = np.asarray(bs_position, dtype=np.float64)
        fractions = np.linspace(0.0, 1.0, corridor_samples, dtype=np.float64)
        corridor_xy = (
            bs_position[None, None, :2]
            + fractions[None, :, None]
            * (positions[:, None, :2] - bs_position[None, None, :2])
        )
        terrain_height = self.sample(self.height, corridor_xy)
        density = self.sample(self.log_density, corridor_xy)
        direct_height = (
            bs_position[2]
            + fractions[None, :] * (positions[:, None, 2] - bs_position[2])
        )
        clearance = terrain_height - direct_height
        offsets = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        grid_x, grid_y = np.meshgrid(offsets, offsets, indexing="xy")
        patch_xy = positions[:, None, :2] + np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)[None]
        patch_height = self.sample(self.height, patch_xy)
        patch_density = self.sample(self.log_density, patch_xy)
        relative = positions[:, :2] - bs_position[:2]
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        direction = relative / np.maximum(distance, 1e-6)
        summary = np.column_stack(
            (
                distance / 300.0,
                direction,
                clearance.max(axis=1) / 50.0,
                (clearance > 0).mean(axis=1),
                terrain_height.max(axis=1) / 50.0,
                density.mean(axis=1) / 10.0,
            )
        )
        return np.concatenate(
            (
                summary,
                terrain_height / 50.0,
                density / 10.0,
                clearance / 50.0,
                patch_height / 50.0,
                patch_density / 10.0,
            ),
            axis=1,
        ).astype(np.float32)
