import argparse
import multiprocessing
from pathlib import Path
from collections.abc import Callable

import torch

from mcunet.tinynas import SampleManager
from mcunet.tinynas.searchspace import configurations, Model
from mcunet.tinynas.searchspace.model import build_model
from mcunet.tinynas.configurations.mnasnetplus import MnasNetPlus

from mcunet.shim.runner import ShimSubprocessManager


def _measure_wrapper(
    classes: int, runner: ShimSubprocessManager
) -> Callable[[Model, float, int], tuple[int, int, int]]:
    def measure(
        model: Model, width_mult: float, resolution: int
    ) -> tuple[int, int, int]:
        net = build_model(model, classes)
        flops = net.flops(resolution, device=torch.device("cuda:0"))

        flash, sram = runner.memory_footprint(model, classes, resolution)

        return flops, flash, sram

    return measure


def _func(
    samples: int, spaces: list[MnasNetPlus], classes: int, path: Path
) -> None:
    manager = SampleManager(spaces)
    manager.sample(samples)

    runner = ShimSubprocessManager(2)
    measure = _measure_wrapper(classes, runner)

    for space, results in manager.apply(measure):
        name = space.__class__.__name__.lower()
        width_mult = str(space.width_mult)[:3].replace(".", "_")

        csv = f"{name}-{width_mult}-{space.resolution}-{classes}.csv"
        with open(path / csv, "w") as f:
            f.write("flops,flash,sram\n")

            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]}\n")

    runner.stop()


def _spaces_as_chunks(chunks: int) -> list[list[MnasNetPlus]]:
    spaces = [MnasNetPlus(a, b) for a, b in configurations()]

    chunked: list[list[MnasNetPlus]] = []

    cut = len(spaces)
    while cut % chunks != 0:
        cut -= 1

    n = int(cut / chunks)
    for i in range(0, cut - n, n):
        chunked.append(spaces[i:i + n])

    chunked.append(spaces[cut - n:])

    return chunked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples", help="sample size per search space", type=int, default=1000
    )
    parser.add_argument(
        "classes", help="classes of all models", type=int
    )
    parser.add_argument(
        "results", help="path to store results", type=str
    )
    parser.add_argument(
        "-c", "--cores", help="number of cpu cores to use", type=int, default=1
    )
    args = parser.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    chunks = _spaces_as_chunks(args.cores)

    processes: list[multiprocessing.Process] = []
    ctx = multiprocessing.get_context("spawn")
    for core in range(args.cores):
        p = ctx.Process(
            target=_func,
            args=(args.samples, chunks[core], args.classes, results,)
        )
        p.start()

        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
