import time
import torch
import pysdf
import trimesh
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from misc import Network, EvaluationResult


class Evaluator:
    def __init__(self, model_pth: Path, mesh_pth: Path, points: int):
        self.__points = points

        saved_model = torch.load(model_pth, map_location='cuda')

        self.__model: Network = Network().cuda()
        self.__model.load_state_dict(saved_model['model'], strict=False)

        self.__mesh: trimesh.Trimesh = self.__scale_mesh(trimesh.load(mesh_pth))

        self.__sdf_fn: pysdf.SDF =  pysdf.SDF(self.__mesh.vertices, self.__mesh.faces)

    @staticmethod
    def __scale_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        center = mesh.bounds.mean(axis=0)
        scale = 2 / np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) * 0.95
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        return mesh
    
    def __measure_model_size(self) -> float:
        return sum((p.numel() * p.element_size() for p in self.__model.parameters())) / (1024 ** 2)
    
    def evaluate(self) -> EvaluationResult:
        points_surface = self.__mesh.sample(self.__points).astype(np.float32)
        points_surface += np.random.normal(0, 1e-2, points_surface.shape).astype(np.float32)

        points_volume = np.random.uniform(-1, 1, (self.__points, 3)).astype(np.float32)

        with torch.no_grad():
            sdfs_surface = self.__model(torch.tensor(points_surface, device='cuda'))
            sdfs_volume = self.__model(torch.tensor(points_volume, device='cuda'))

        random_points = torch.rand((1000000, 3), device='cuda', dtype=torch.float32) * 2 - 1

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                self.__model(random_points)
        torch.cuda.synchronize()
        end = time.perf_counter()

        batch_time = (end - start) / 10
        per_point_time = batch_time / random_points.shape[0]

        return EvaluationResult(
            model_size=self.__measure_model_size(),
            quality_surface=f1_score(self.__sdf_fn(points_surface) > 0, sdfs_surface.cpu() < 0, average='weighted'),
            quality_volume=f1_score(self.__sdf_fn(points_volume) > 0, sdfs_volume.cpu() < 0, average='weighted'),
            time_per_batch=batch_time * 1000,
            time_per_point=per_point_time * 1e9,
        )
