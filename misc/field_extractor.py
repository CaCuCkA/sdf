import torch
import numpy as np


class FieldExtractor:
    def __init__(self, bound_min: torch.FloatTensor, bound_max: torch.FloatTensor, resolution: int):
        self.__bound_min = bound_min
        self.__bound_max = bound_max
        self.__resolution = resolution

    def extract_fields(self, query_func):
        n = 16
        x_space = torch.linspace(self.__bound_min[0], self.__bound_max[0], self.__resolution).split(n)
        y_space = torch.linspace(self.__bound_min[1], self.__bound_max[1], self.__resolution).split(n)
        z_space = torch.linspace(self.__bound_min[2], self.__bound_max[2], self.__resolution).split(n)

        u = np.zeros([self.__resolution, self.__resolution, self.__resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(x_space):
                for yi, ys in enumerate(y_space):
                    for zi, zs in enumerate(z_space):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * n: xi * n + len(xs), yi * n: yi * n + len(ys), zi * n: zi * n + len(zs)] = val
        return u
    
