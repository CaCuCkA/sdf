import torch
from tqdm import tqdm
from pathlib import Path
from typing import Generator, Tuple
from argparse import ArgumentParser

from misc import Network, MeshGenerator


def get_input_output_data(root_path: str, output_root: str) -> Generator[Tuple[Path, Path], None, None]:
    root = Path(root_path)
    output_root_path = Path(output_root)
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            best_pth_path = subdir / 'best.pth'
            if best_pth_path.exists():
                mesh_output_path = output_root_path / f"{subdir.name}.ply"
                yield (best_pth_path, mesh_output_path)


def count_model_directories(path: Path) -> int:
    count = 0
    for subdir in path.iterdir():
        if subdir.is_dir() and subdir.name.isdigit() and (subdir / 'best.pth').exists():
            count += 1
    return count


def get_mesh(input_path: Path, output_path: Path, resolution: int) -> None:
    saved = torch.load(input_path, map_location='cuda')
    model = Network().cuda()
    model.load_state_dict(saved['model'], strict=False)

    mesh_generator = MeshGenerator(model, resolution)
    mesh = mesh_generator.generate_mesh()
    mesh.export(output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--output', type=str, required=True, help='Path to the output .ply files')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution')
    options = parser.parse_args()

    model_path = Path(options.model)
    total_models = count_model_directories(model_path)
    data_generator = get_input_output_data(options.model, options.output)

    for input_path, output_path in tqdm(data_generator, total=total_models, desc="Processing Models"):
        get_mesh(input_path, output_path, options.resolution)


if __name__ == "__main__":
    main()
