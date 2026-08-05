from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from physical_ai.map_encoder import read_open3d_binary_ply


@dataclass
class AdvancedMapRaster:
    """Normal-aware raster used to build learned Physical-AI map descriptors."""

    minimum_xy: np.ndarray
    resolution: float
    height: np.ndarray
    mean_height: np.ndarray
    height_std: np.ndarray
    log_density: np.ndarray
    wall_log_density: np.ndarray
    horizontal_log_density: np.ndarray
    wall_xx: np.ndarray
    wall_xy: np.ndarray
    wall_yy: np.ndarray

    @classmethod
    def from_point_cloud(
        cls,
        path: str | Path,
        minimum_xy: np.ndarray,
        maximum_xy: np.ndarray,
        resolution: float = 2.0,
    ) -> "AdvancedMapRaster":
        points = read_open3d_binary_ply(path)
        minimum_xy = np.asarray(minimum_xy, dtype=np.float64)
        maximum_xy = np.asarray(maximum_xy, dtype=np.float64)
        shape_xy = np.ceil((maximum_xy - minimum_xy) / resolution).astype(int) + 1
        shape = (shape_xy[1], shape_xy[0])
        height = np.zeros(shape, np.float32)
        count = np.zeros(shape, np.int32)
        wall_count = np.zeros(shape, np.int32)
        horizontal_count = np.zeros(shape, np.int32)
        sum_z = np.zeros(shape, np.float64)
        sum_z2 = np.zeros(shape, np.float64)
        sum_xx = np.zeros(shape, np.float64)
        sum_xy = np.zeros(shape, np.float64)
        sum_yy = np.zeros(shape, np.float64)
        chunk = 250_000
        for start in range(0, len(points), chunk):
            block = np.asarray(points[start : start + chunk])
            xyz, normal = block[:, :3], block[:, 3:6]
            cell = np.floor((xyz[:, :2] - minimum_xy) / resolution).astype(np.int64)
            valid = (
                (cell[:, 0] >= 0)
                & (cell[:, 0] < shape_xy[0])
                & (cell[:, 1] >= 0)
                & (cell[:, 1] < shape_xy[1])
            )
            x, y = cell[valid, 0], cell[valid, 1]
            z = xyz[valid, 2].astype(np.float32)
            n = normal[valid].astype(np.float32)
            np.maximum.at(height, (y, x), z)
            np.add.at(count, (y, x), 1)
            np.add.at(sum_z, (y, x), z)
            np.add.at(sum_z2, (y, x), z * z)
            wall = np.abs(n[:, 2]) < 0.4
            horizontal = np.abs(n[:, 2]) > 0.8
            wy, wx, wn = y[wall], x[wall], n[wall]
            np.add.at(wall_count, (wy, wx), 1)
            np.add.at(horizontal_count, (y[horizontal], x[horizontal]), 1)
            np.add.at(sum_xx, (wy, wx), wn[:, 0] * wn[:, 0])
            np.add.at(sum_xy, (wy, wx), wn[:, 0] * wn[:, 1])
            np.add.at(sum_yy, (wy, wx), wn[:, 1] * wn[:, 1])
        safe_count = np.maximum(count, 1)
        mean = sum_z / safe_count
        variance = np.maximum(sum_z2 / safe_count - mean * mean, 0.0)
        safe_wall = np.maximum(wall_count, 1)
        return cls(
            minimum_xy=minimum_xy,
            resolution=float(resolution),
            height=height,
            mean_height=mean.astype(np.float32),
            height_std=np.sqrt(variance).astype(np.float32),
            log_density=np.log1p(count).astype(np.float32),
            wall_log_density=np.log1p(wall_count).astype(np.float32),
            horizontal_log_density=np.log1p(horizontal_count).astype(np.float32),
            wall_xx=(sum_xx / safe_wall).astype(np.float32),
            wall_xy=(sum_xy / safe_wall).astype(np.float32),
            wall_yy=(sum_yy / safe_wall).astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> "AdvancedMapRaster":
        raw = np.load(path)
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})

    def sample(self, image: np.ndarray, xy: np.ndarray) -> np.ndarray:
        cell = (np.asarray(xy) - self.minimum_xy) / self.resolution
        coordinates = np.stack((cell[..., 1], cell[..., 0]), axis=0)
        return map_coordinates(image, coordinates, order=1, mode="nearest")

    def context_features(
        self,
        positions: np.ndarray,
        bs_position: np.ndarray,
        corridor_samples: int = 64,
    ) -> np.ndarray:
        positions = np.asarray(positions, np.float64)
        bs = np.asarray(bs_position, np.float64)
        relative = positions[:, :2] - bs[:2]
        distance = np.linalg.norm(relative, axis=1)
        direction = relative / np.maximum(distance[:, None], 1e-6)
        perpendicular = np.column_stack((-direction[:, 1], direction[:, 0]))

        fractions = np.linspace(0.02, 0.98, corridor_samples)
        center = bs[None, None, :2] + fractions[None, :, None] * relative[:, None]
        offsets = np.array([-12.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0])
        corridor = center[:, None] + offsets[None, :, None, None] * perpendicular[:, None, None]
        terrain = self.sample(self.height, corridor)
        # Keep the direct path as (point, sample).  An extra singleton point
        # axis here would broadcast clearance to (point, point, offset, sample)
        # and make feature extraction quadratic in the batch size.
        direct = bs[2] + fractions[None, :] * (
            positions[:, None, 2] - bs[2]
        )
        clearance = terrain - direct[:, None, :]
        blocked = clearance > 0.5
        positive = np.maximum(clearance, 0.0)
        transitions = np.abs(np.diff(blocked.astype(np.float32), axis=-1)).mean(-1)
        peak_fraction = fractions[np.argmax(clearance, axis=-1)]
        middle_weight = np.sin(np.pi * fractions)
        middle_weight /= middle_weight.mean()
        corridor_summary = np.stack(
            (
                clearance.max(-1) / 50.0,
                positive.mean(-1) / 50.0,
                blocked.mean(-1),
                (positive * middle_weight).mean(-1) / 50.0,
                peak_fraction,
                transitions,
            ),
            axis=-1,
        ).reshape(len(positions), -1)

        wall_density = self.sample(self.wall_log_density, corridor) / 10.0
        xx = self.sample(self.wall_xx, corridor)
        xy = self.sample(self.wall_xy, corridor)
        yy = self.sample(self.wall_yy, corridor)
        dx = direction[:, 0, None, None]
        dy = direction[:, 1, None, None]
        normal_alignment = dx * dx * xx + 2.0 * dx * dy * xy + dy * dy * yy
        corridor_material = np.stack(
            (
                wall_density.mean(-1),
                wall_density.max(-1),
                normal_alignment.mean(-1),
                self.sample(self.height_std, corridor).mean(-1) / 20.0,
            ),
            axis=-1,
        ).reshape(len(positions), -1)

        sectors = 16
        angles = np.linspace(0.0, 2.0 * np.pi, sectors, endpoint=False)
        unit = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        radii = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
        ring_xy = positions[:, None, None, :2] + radii[None, :, None, None] * unit[None, None]
        ring_height = self.sample(self.height, ring_xy) - positions[:, None, None, 2]
        ring_wall = self.sample(self.wall_log_density, ring_xy) / 10.0
        ring_horizontal = self.sample(self.horizontal_log_density, ring_xy) / 10.0
        ring_roughness = self.sample(self.height_std, ring_xy) / 20.0
        ring_summary = np.stack(
            (
                ring_height.mean(-1) / 50.0,
                ring_height.max(-1) / 50.0,
                ring_height.std(-1) / 50.0,
                ring_wall.mean(-1),
                ring_wall.max(-1),
                ring_horizontal.mean(-1),
                ring_roughness.mean(-1),
            ),
            axis=-1,
        ).reshape(len(positions), -1)

        # Rotate angular profiles so sector zero points from the terminal toward the BS.
        terminal_angle = np.arctan2(-direction[:, 1], -direction[:, 0])
        sector_shift = np.rint(terminal_angle / (2.0 * np.pi) * sectors).astype(int)
        skyline_height = np.empty((len(positions), 3, sectors), np.float32)
        skyline_wall = np.empty_like(skyline_height)
        for row, shift in enumerate(sector_shift):
            skyline_height[row] = np.roll(ring_height[row, 2:] / 50.0, -shift, axis=-1)
            skyline_wall[row] = np.roll(ring_wall[row, 2:], -shift, axis=-1)

        # Smoothed local fields capture neighborhood topology at scales larger than one cell.
        smooth_features = []
        for sigma_m in (4.0, 8.0, 16.0, 32.0):
            sigma = sigma_m / self.resolution
            smooth_height = gaussian_filter(self.height, sigma=sigma, mode="nearest")
            smooth_wall = gaussian_filter(self.wall_log_density, sigma=sigma, mode="nearest")
            smooth_features.extend(
                (self.sample(smooth_height, positions[:, :2]) / 50.0,
                 self.sample(smooth_wall, positions[:, :2]) / 10.0)
            )
        smooth = np.stack(smooth_features, axis=-1)
        result = np.concatenate(
            (
                corridor_summary,
                corridor_material,
                ring_summary,
                skyline_height.reshape(len(positions), -1),
                skyline_wall.reshape(len(positions), -1),
                smooth,
            ),
            axis=1,
        )
        return np.nan_to_num(result, copy=False).astype(np.float32)


ADVANCED_CONTEXT_SLICES = {
    "corridor2": slice(0, 42),
    "material": slice(42, 70),
    "endpoint": slice(70, 105),
    "skyline": slice(105, 201),
    "multiscale": slice(201, 209),
    "advanced": slice(0, 209),
}
