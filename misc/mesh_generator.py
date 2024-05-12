import torch

from trimesh import Trimesh
from mcubes import marching_cubes

from misc import Network, FieldExtractor


class MeshGenerator:
    def __init__(self, model: Network, resolution: int, threshold: float = 0):
        self.__model = model
        self.__resolution = resolution
        self.__threshold = threshold

    def generate_mesh(self) -> Trimesh:
        def query_func(pts):
            pts = pts.to('cuda')
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    sdfs = self.__model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-1, -1, -1])
        bounds_max = torch.FloatTensor([1, 1, 1])
        extractor = FieldExtractor(bounds_min, bounds_max, self.__resolution)
        
        u = extractor.extract_fields(query_func)
        vertices, triangles = marching_cubes(u, self.__threshold)

        b_max_np = bounds_max.detach().cpu().numpy()
        b_min_np = bounds_min.detach().cpu().numpy()
        vertices = vertices / (self.__resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

        return Trimesh(vertices, triangles, process=False)
    