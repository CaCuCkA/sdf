import os
from tqdm import tqdm
from pathlib import Path
from pandas import DataFrame
from typing import Generator, Any
from argparse import ArgumentParser

from misc import Evaluator


def iterate_all_trained_models(path: Path) -> Generator[Path, None, None]:
    for entry in os.listdir(path):
        entry_path = path / entry
        if entry_path.is_dir():
            yield entry_path


def main():
    parser = ArgumentParser()
    parser.add_argument('--models', type=Path, required=True, help='Path to the trained models')
    parser.add_argument('--meshes', type=Path, required=True, help='Path to the original objects')
    parser.add_argument('--output', type=Path, required=True, help='Output csv path')
    parser.add_argument('--points', type=int, default=10000, help='Number of points')

    options = parser.parse_args()
    models = list(iterate_all_trained_models(options.models))

    results = {
        model.stem: Evaluator(model / "best.pth", options.meshes / f"{model.stem}.obj", options.points).evaluate()
        for model in tqdm(models)
    }
    results_data = [
        [model, res.model_size, res.quality_surface, res.quality_volume, res.time_per_batch, res.time_per_point]
        for model, res in results.items()
    ]

    df = DataFrame(results_data, columns=["Model number", "Model size (MB)", "Quality (surface)", "Quality (volume)",
                                          "Time per 100k batch (ms)", "Time per point (ns)"])
    df.sort_values("Model number", inplace=True)
    df.to_csv(options.output, index=False, float_format='%.4f')

    print(f"Model size: {df['Model size (MB)'].mean():.4f} MB")
    print(f"Mean quality near surface: {df['Quality (surface)'].mean():.4f}")
    print(f"Mean quality on the bounding volume: {df['Quality (volume)'].mean():.4f}")
    print(f"Mean time per 100k batch: {df['Time per 100k batch (ms)'].mean():.4f} ms")
    print(f"Mean time per point: {df['Time per point (ns)'].mean():.4f} ns")


if __name__ == "__main__":
    main()
