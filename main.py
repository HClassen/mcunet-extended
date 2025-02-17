import re
import sys
import json
import argparse
from pathlib import Path
from shutil import rmtree
from collections.abc import Callable

import torch

from mcunet.tinynas import SearchManager, SaveEvolutionCtx
from mcunet.tinynas.oneshot import SuperNet, OneShotNet
from mcunet.tinynas.searchspace import Model
from mcunet.tinynas.datasets import transform_resize, gtsrb
from mcunet.tinynas.utils import ConsoleLogger, FileLogger
from mcunet.tinynas.configurations.mnasnetplus import (
    mutator,
    super_choice_blocks_wrapper,
    LastLayerSharer,
    MnasNetPlus
)
from mcunet.tinynas.configurations.mnasnetplus.share import full

from mcunet.shim.runner import (
    memory_footprint,
    start_shim_runner,
    stop_shim_runner
)


def _parse_memory(memory: str) -> int:
    if memory == "inf":
        return sys.maxsize

    lower = memory.lower()
    match = re.match(r"^[0-9]+((k|m|g)?b)?$", lower)
    if match is None:
        raise RuntimeError(f"invalid memory size {memory}")

    if lower[-1].isdigit():
        return int(memory)

    unit_begin = -2 if lower[-2].isalpha() else -1

    value = lower[:unit_begin]
    unit = lower[unit_begin:]

    units = {"b": 1, "kb": 1_000, "mb": 1_000_000, "gb": 1_000_000_000}
    return int(value) * units[unit]


def _save_manager(path: Path, manager: SearchManager) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True)

    state = manager.to_dict()
    weights = state["weights"]
    del state["weights"]

    with open(path / "state.json", "w") as f:
        json.dump(state, f)

    with open(path / "weights.pth", "wb") as f:
        torch.save(weights, f)

def _save_top(path: Path, k: int, topk: list[tuple[Model, float]]) -> None:
    for i, (candidate, accuracy) in enumerate(topk):
        directory = path / f"top-{i}"
        directory.mkdir(parents=True, exist_ok=True)

        with open (directory / "meta.json", "w") as f:
            json.dump({"model": candidate.to_dict(), "accuracy": accuracy}, f)

        if i >= k:
            break


def _fitness_wrapper(
    classes: int, resolution: int, max_flash: int, max_sram: int
) -> Callable[[Model], bool]:
    def fitness(model: Model) -> bool:
        flash, sram = memory_footprint(model, classes, resolution)

        return flash <= max_flash and sram <= max_sram

    return fitness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="path to the root of the GTSRB data set",
        type=str
    )
    parser.add_argument(
        "results",
        help="path to store the results of the NAS",
        type=str
    )
    parser.add_argument(
        "-r", "--resolution",
        required=True,
        help="the resolution of the images",
        type=int
    )
    parser.add_argument(
        "-w", "--width-mult",
        required=True,
        help="the width multiplier",
        type=float
    )
    parser.add_argument(
        "-f", "--flash",
        default="inf",
        help="the flash limitation",
        type=str
    )
    parser.add_argument(
        "-s", "--sram",
        default="inf",
        help="the sram limitation",
        type=str
    )
    parser.add_argument(
        "-l", "--logging",
        help="path to store logging output or stdout",
        type=str,
        default="stdout"
    )
    args = parser.parse_args()

    path_results = Path(args.results)
    if not path_results.exists():
        path_results.mkdir(parents=True)

    path_ds = Path(args.dataset)
    if not path_ds.exists():
        gtsrb.get(path_ds)

    ds = gtsrb.GTSRBDataset(
        path_ds, "train", transform=transform_resize(args.resolution)
    )

    space = MnasNetPlus(args.width_mult, args.resolution)

    block_constructor = super_choice_blocks_wrapper(
        full.Conv2dSharer, full.DWConv2dSharer, full.BDWRConv2dSharer
    )
    last_sharer = LastLayerSharer
    supernet = SuperNet(
        block_constructor,
        last_sharer,
        ds.classes,
        args.width_mult,
        initialize_weights=False
    )

    logger = ConsoleLogger() if args.logging == "stdout" \
             else FileLogger(args.logging)

    manager = SearchManager(space, ds, supernet, logger)
    manager.train(
        epochs=300,
        batch_size=256,
        models_per_batch=1,
        warm_up=True,
        warm_up_ctx=full.FullWarmUpCtx(),
        warm_up_epochs=25,
        warm_up_batch_size=256,
        device=torch.device("cuda:0")
    )
    _save_manager(path_results / "supernet", manager)

    start_shim_runner()

    max_flash = _parse_memory(args.flash)
    max_sram = _parse_memory(args.sram)
    fitness = _fitness_wrapper(ds.classes, args.resolution, max_flash, max_sram)
    top = manager.evolution(
        mutator=mutator,
        fitness=fitness,
        ctx=SaveEvolutionCtx(path_results / "intermediate", 2, first=True),
        device=torch.device("cuda:0")
    )

    _save_top(path_results / "evolution", 5, top)

    stop_shim_runner()


if __name__ == "__main__":
    main()
