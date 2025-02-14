import argparse
import multiprocessing
from pathlib import Path
from collections.abc import Callable

import torch

from mcunet.tinynas import SampleManager
from mcunet.tinynas.searchspace import configurations, Model, SearchSpace
from mcunet.tinynas.searchspace.model import build_model
from mcunet.tinynas.configurations.mnasnetplus import MnasNetPlus

from mcunet.shim.runner import ShimSubprocessManager


def _measure_wrapper(
    samples: int,
    classes: int,
    runner: ShimSubprocessManager,
    meta: dict[MnasNetPlus, tuple[Path, int]]
) -> Callable[[SearchSpace, Model], tuple[int, int, int]]:
    def measure(
        space: SearchSpace, model: Model
    ) -> tuple[int, int, int]:
        t = meta[space]
        if t[1] >= samples + 1:
            return -1, -1, -1

        net = build_model(model, classes)
        flops = net.flops(space.resolution, device=torch.device("cpu"))

        flash, sram = runner.memory_footprint(model, classes, space.resolution)

        return flops, flash, sram

    return measure


def _space_to_csv(space: MnasNetPlus, classes: int) -> str:
    name = space.__class__.__name__.lower()
    width_mult = str(space.width_mult)[:3].replace(".", "_")

    return f"{name}-{width_mult}-{space.resolution}-{classes}.csv"


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0

    with open(path, "r") as f:
        count = len(f.readlines())

    return count


def _func(
    samples: int, spaces: list[MnasNetPlus], classes: int, path: Path
) -> None:
    meta: dict[SearchSpace, list[Path, int]] = {}
    for space in spaces:
        csv = _space_to_csv(space, classes)
        path_csv = path / csv
        lines = _count_lines(path_csv)

        meta[space] = [path_csv, lines]

    manager = SampleManager(spaces)
    manager.sample(samples)

    runner = ShimSubprocessManager(2)
    measure = _measure_wrapper(samples, classes, runner, meta)

    for space, results in manager.apply(measure):
        t = meta[space]
        if t[1] >= samples + 1:
            continue

        with open(t[0], "a") as f:
            if t[1] == 0:
                f.write("flops,flash,sram\n")
                t[1] = 1

            f.write(f"{results[0]},{results[1]},{results[2]}\n")
            t[1] += 1

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
