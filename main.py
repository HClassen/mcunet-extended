import re
import sys
import json
import argparse
from typing import Any
from pathlib import Path
from shutil import rmtree
from collections.abc import Callable

import torch

from mcunet.tinynas import SearchManager, SaveTrainCtx
from mcunet.tinynas.oneshot import SuperNet, OneShotNet
from mcunet.tinynas.searchspace import Model
from mcunet.tinynas.datasets import transform_resize, CustomDataset, imagenet, gtsrb
from mcunet.tinynas.utils import Logger, ConsoleLogger, FileLogger
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

    units = {"b": 1, "kb": 1_024, "mb": 1_024**2, "gb": 1_024**3}
    return int(value) * units[unit]


def _save_manager(path: Path, manager: SearchManager) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir()

    state = manager.to_dict()
    weights = state["weights"]
    del state["weights"]

    with open(path / "state.json", "w") as f:
        json.dump(state, f)

    with open(path / "weights.pth", "wb") as f:
        torch.save(weights, f)

def _save_top(path: Path, k: int, topk: tuple[tuple[OneShotNet, float]]) -> None:
    for i, (candidate, accuracy) in enumerate(topk):
        directory = path / f"top-{i}"
        directory.mkdir(exist_ok=True)

        model = candidate.to_model()
        with open (directory / "meta.json", "w") as f:
            json.dump({"model": model.to_dict(), "accuracy": accuracy}, f)

        with open(directory / "weights.pth", "wb") as f:
            torch.save(candidate.state_dict(), f)

        if i >= k:
            break


def _fitness_wrapper(
    classes: int, resolution: int, max_flash: int, max_sram: int
) -> Callable[[Model], bool]:
    def fitness(model: Model) -> bool:
        flash, sram = memory_footprint(model, classes, resolution)

        return flash <= max_flash and sram <= max_sram

    return fitness


def _load_ds(
    name: str, 
    path: str, 
    split: str,
    transform: Callable[[Any], Any] | None,
    from_json: bool
) -> CustomDataset:
    match name:
        case "ilsvrc":
            if not from_json:
                return imagenet.ImageNetDataset(path, split, transform)
            
            with open(path, "r") as f:
                config = json.load(f)
            return imagenet.ImageNetDataset.from_dict(config, transform)
        case "gtsrb":
            return gtsrb.GTSRBDataset(path, split, transform)
        case _:
            raise RuntimeError(f"unknown data set {name}")
        

def _common(
    width_mult: float, 
    resolution: int, 
    classes: int, 
    initialize_weights: bool, 
    logging: str
) -> tuple[MnasNetPlus, SuperNet, Logger]:

    space = MnasNetPlus(width_mult, resolution)

    block_constructor = super_choice_blocks_wrapper(
        full.Conv2dSharer, full.DWConv2dSharer, full.BDWRConv2dSharer
    )
    last_sharer = LastLayerSharer
    supernet = SuperNet(
        block_constructor,
        last_sharer,
        classes,
        width_mult,
        initialize_weights=initialize_weights
    )

    logger = ConsoleLogger() if logging == "stdout" else FileLogger(logging)

    return space, supernet, logger


def _train(args: argparse.Namespace) -> None:
    path_results = Path(args.results)
    if not path_results.exists():
        path_results.mkdir()

    ds = _load_ds(
        args.dataset, 
        args.path, 
        "train", 
        transform_resize(args.resolution),
        args.ds_json
    )

    space, supernet, logger = _common(
        args.width_mult, args.resolution, ds.classes, True, args.logger
    )

    manager = SearchManager(space, ds, supernet, logger)

    if args.save_every > 0:
        train_ctx = SaveTrainCtx(path_results / "intermediate", args.save_every)
    else:
        train_ctx = None

    manager.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        models_per_batch=args.models_per_batch,
        warm_up=False,
        train_ctx=train_ctx,
        device=torch.device("cuda:0")
    )
    _save_manager(path_results / "supernet", manager)


def _evo(args: argparse.Namespace) -> None:
    path_results = Path(args.results)
    if not path_results.exists():
        path_results.mkdir()

    ds = _load_ds(
        args.dataset, 
        args.path, 
        "train", 
        transform_resize(args.resolution),
        args.ds_json
    )

    space, supernet, logger = _common(
        args.width_mult, args.resolution, ds.classes, False, args.logger
    )

    with open(Path(args.manager) / "state.json", "r") as f:
        state = json.load(f)

    with open(Path(args.manager) / "weights.pth", "rb") as f:
        state["weights"] = torch.load(f, weights_only=True)

    manager = SearchManager.from_dict(state, space, ds, supernet)
    manager._logger = logger

    start_shim_runner()

    max_flash = _parse_memory(args.flash)
    max_sram = _parse_memory(args.sram)
    fitness = _fitness_wrapper(ds.classes, args.resolution, max_flash, max_sram)
    top = manager.evolution(
        topk=args.topk,
        valid_batch_size=args.batch_size,
        calib_batch_size=args.calib_batch_size,
        iterations=args.iterations,
        population_size=args.population,
        mutator=mutator, 
        fitness=fitness, 
        device=torch.device("cuda:0")
    )

    stop_shim_runner()

    _save_top(path_results / "evolution", args.save_top, top)


def _add_common_args(*args) -> None:
    for parser in args:
        parser.add_argument(
            "--ds-json",
            help="load the data set from a JSON file instead of a folder",
            action="store_true"
        )
        parser.add_argument(
            "dataset",
            help="the data set to use",
            type=str,
            choices=["ilsvrc", "gtsrb"]
        )
        parser.add_argument(
            "path",
            help="the path to the data set",
            type=str
        )
        parser.add_argument(
            "results",
            help="path to store the results of the NAS",
            type=str
        )


def main() -> None:
    parser = argparse.ArgumentParser()
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

    sub = parser.add_subparsers()
    
    train = sub.add_parser(
        "train", 
        help="performs only the training of the supernet"
    )
    train.add_argument(
        "--save-every",
        help="save the intermediate state of the training every n epoch",
        type=int,
        default=0
    )
    train.add_argument(
        "--epochs",
        help="the number of epochs to train",
        type=int,
        default=120
    )
    train.add_argument(
        "--batch-size",
        help="the batch size used for training",
        type=int,
        default=1024
    )
    train.add_argument(
        "--models-per-batch",
        help="the number of madels smapled per batch of training data",
        type=int,
        default=1
    )
    train.set_defaults(func=_train)

    evo = sub.add_parser(
        "evo", 
        help="performs only the evolution search"
    )
    evo.add_argument(
        "--flash",
        help="the flash limitation",
        type=str,
        default="inf"
    )
    evo.add_argument(
        "--sram",
        help="the sram limitation",
        type=str,
        default="inf"
    )
    evo.add_argument(
        "--topk",
        help="the topk models to keep",
        type=int,
        default=20
    )
    evo.add_argument(
        "--save-topk",
        help="save the first n from topk",
        type=int,
        default=5
    )
    evo.add_argument(
        "--batch-size",
        help="the batch size used for validation",
        type=int,
        default=1024
    )
    evo.add_argument(
        "--calib-batch-size",
        help="the batch size used for calibration",
        type=int,
        default=64
    )
    evo.add_argument(
        "--population",
        help="the population size",
        type=int,
        default=100
    )
    evo.add_argument(
        "--iterations",
        help="the number of iterations",
        type=int,
        default=30
    )
    evo.add_argument(
        "manager",
        help="the path to the manager",
        type=str
    )
    evo.set_defaults(func=_evo)

    _add_common_args(train, evo)

    parser.parse_args()


if __name__ == "__main__":
    main()
