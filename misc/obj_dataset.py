import numpy as np

from torch.utils.data import Dataset

import trimesh
import pysdf


class SDFDataset(Dataset):
    def __init__(self, path: str, size: int = 100, batch_size: int = 2 ** 18):
        super().__init__()

        self.__mesh = trimesh.load(path, force='mesh')

        v_min = self.__mesh.vertices.min(0)
        v_max = self.__mesh.vertices.max(0)
        v_center = (v_min + v_max) / 2
        v_scale = 2 / np.sqrt(np.sum((v_max - v_min) ** 2)) * 0.95

        self.__mesh.vertices = (self.__mesh.vertices - v_center[None, :]) * v_scale

        self.__sdf_fn = pysdf.SDF(self.__mesh.vertices, self.__mesh.faces)

        self._batch_size = batch_size
        assert self._batch_size % 8 == 0, "batch size must be divisible by 8."

        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, _):
        sdfs = np.zeros((self._batch_size, 1))
        points_surface = self.__mesh.sample(self._batch_size * 7 // 8)
        points_surface[self._batch_size // 2:] += 0.01 * np.random.randn(self._batch_size * 3 // 8, 3)
        points_uniform = np.random.rand(self._batch_size // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self._batch_size // 2:] = -self.__sdf_fn(points[self._batch_size // 2:])[:, None].astype(np.float32)

        return {
            'sdfs': sdfs,
            'points': points,
        }
    