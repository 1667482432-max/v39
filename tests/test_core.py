from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from physical_ai.data import RoundData
from physical_ai.advanced_map import ADVANCED_CONTEXT_SLICES, AdvancedMapRaster
from physical_ai.features import SpectralFeatureLayout, spectral_targets_from_features
from physical_ai.neighbors import distance_weights, nearest_neighbors
from physical_ai.spectral import replace_fourier_magnitude
from physical_ai.spatial import metric_embeddings
from physical_ai.transductive import transductive_graph_features


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = RoundData(ROOT)

    def test_authoritative_shapes(self) -> None:
        self.assertEqual(self.data.train_positions.shape, (2000, 3))
        self.assertEqual(self.data.test_positions.shape, (500, 3))
        self.assertEqual(self.data.train_channels.shape, (2000, 256, 4, 192))
        self.assertEqual(self.data.train_channels.dtype, np.complex64)

    def test_neighbor_weights(self) -> None:
        indices, distances = nearest_neighbors(
            np.asarray(self.data.test_positions[:5]), np.asarray(self.data.train_positions), 16
        )
        weights = distance_weights(distances, power=2.0)
        self.assertEqual(indices.shape, (5, 16))
        np.testing.assert_allclose(weights.sum(1), 1.0, atol=1e-7)
        self.assertTrue(np.all(weights >= 0))

    def test_compact_target_energy_is_compatible(self) -> None:
        layout = SpectralFeatureLayout.from_dimensions(self.data.dims)
        features = torch.rand(layout.total_size)
        pas, pdp = spectral_targets_from_features(features, self.data.dims)
        self.assertEqual(tuple(pas.shape), self.data.dims.channel_shape)
        self.assertEqual(tuple(pdp.shape), self.data.dims.channel_shape)
        torch.testing.assert_close(pas.sum(), pdp.sum(), rtol=1e-5, atol=1e-5)

    def test_magnitude_projection(self) -> None:
        generator = torch.Generator().manual_seed(7)
        channel = torch.complex(torch.randn(8, 3, 12, generator=generator), torch.randn(8, 3, 12, generator=generator))
        target_power = torch.rand(8, 3, 12, generator=generator).clamp_min(1e-3)
        projected = replace_fourier_magnitude(channel, target_power, dim=-1)
        actual = torch.abs(torch.fft.fft(projected, dim=-1, norm="ortho")).square()
        torch.testing.assert_close(actual, target_power, rtol=2e-5, atol=2e-5)

    def test_transductive_zero_alpha_is_identity(self) -> None:
        labeled_positions = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        unlabeled_positions = np.array([[0.5, 0.5], [1.0, 1.0]])
        labeled_features = np.arange(12, dtype=np.float32).reshape(3, 4)
        direct = np.full((2, 4), 3.5, dtype=np.float32)
        prediction = transductive_graph_features(
            labeled_positions,
            unlabeled_positions,
            labeled_features,
            direct,
            k=2,
            alpha=0.0,
        )
        np.testing.assert_array_equal(prediction, direct)

    def test_advanced_map_context_has_batch_independent_width(self) -> None:
        shape = (48, 48)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        height = (0.15 * xx + 0.08 * yy).astype(np.float32)
        ones = np.ones(shape, dtype=np.float32)
        raster = AdvancedMapRaster(
            minimum_xy=np.zeros(2),
            resolution=1.0,
            height=height,
            mean_height=height * 0.8,
            height_std=ones * 0.2,
            log_density=ones,
            wall_log_density=ones * 0.5,
            horizontal_log_density=ones * 0.75,
            wall_xx=ones * 0.6,
            wall_xy=np.zeros(shape, dtype=np.float32),
            wall_yy=ones * 0.4,
        )
        positions = np.array(
            [[14.0, 13.0, 1.5], [23.0, 19.0, 1.5], [31.0, 29.0, 1.5]],
            dtype=np.float32,
        )
        context = raster.context_features(positions, np.array([4.0, 4.0, 12.0]))
        expected_width = ADVANCED_CONTEXT_SLICES["advanced"].stop
        self.assertEqual(context.shape, (len(positions), expected_width))
        self.assertTrue(np.isfinite(context).all())

    def test_advanced_metric_embedding_is_opt_in(self) -> None:
        rng = np.random.default_rng(9)
        positions = rng.normal(size=(12, 3)).astype(np.float32)
        legacy = rng.normal(size=(12, 153)).astype(np.float32)
        advanced = rng.normal(size=(12, 209)).astype(np.float32)
        name = "xy_ctx-material-center-multiscale_s3"
        self.assertNotIn(name, metric_embeddings(positions, legacy))
        embedding = metric_embeddings(
            positions, np.concatenate((legacy, advanced), axis=1)
        )[name]
        self.assertEqual(embedding.shape, (12, 2 + 25 + 25 + 12))
        self.assertTrue(np.isfinite(embedding).all())

    def test_submission_header_if_present(self) -> None:
        path = ROOT / "Round1_Test_Channel.npy"
        if path.exists():
            output = np.load(path, mmap_mode="r")
            self.assertEqual(output.shape, (500, 256, 4, 192))
            self.assertEqual(output.dtype, np.complex64)


if __name__ == "__main__":
    unittest.main()
