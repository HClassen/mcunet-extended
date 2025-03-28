import re
import sys
import json
import argparse
from typing import Any
from pathlib import Path
from shutil import rmtree
from collections.abc import Callable

import torch

from mcunet.tinynas import SearchManager, SaveTrainCtx, SaveEvolutionCtx
from mcunet.tinynas.oneshot import SuperNet
from mcunet.tinynas.searchspace import Model
from mcunet.tinynas.utils import ConsoleLogger, FileLogger
from mcunet.tinynas.instances import (
    Conv2dSharer,
    DWConv2dSharer,
    BDWRConv2dSharer,
    LastLayerSharer
)
from mcunet.tinynas.instances.full import (
    mutator,
    super_choice_blocks_wrapper,
    FullSearchSpace
)

from mcunet.shim.runner import (
    memory_footprint,
    start_shim_runner,
    stop_shim_runner
)

from helper import *


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

def _save_top(path: Path, k: int, top: list[tuple[Model, float]]) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir()

    for i, (model, accuracy) in enumerate(top[:k]):
        with open (path / f"top-{i}.json", "w") as f:
            json.dump({"model": model.to_dict(), "accuracy": accuracy}, f)


def _fitness_wrapper(
    classes: int, resolution: int, max_flash: int, max_sram: int
) -> Callable[[Model], bool]:
    def fitness(model: Model) -> bool:
        flash, sram = memory_footprint(model, classes, resolution)

        return flash <= max_flash and sram <= max_sram

    return fitness


def _common(
    args: argparse.Namespace,
    initialize_weights: bool,
    state: dict[str, Any] | None
) -> tuple[SearchManager, Path]:
    path_results = Path(args.results)
    path_results.mkdir(exist_ok=True, parents=True)

    ds = ds_load(args.dataset, args.path, "train", args.ds_json, args.resolution)
    space = FullSearchSpace(args.width_mult, args.resolution)

    block_constructor = super_choice_blocks_wrapper(
        Conv2dSharer, DWConv2dSharer, BDWRConv2dSharer
    )
    last_sharer = LastLayerSharer
    supernet = SuperNet(
        block_constructor,
        last_sharer,
        ds.classes,
        args.width_mult,
        initialize_weights=initialize_weights
    )

    if args.logger == "stdout":
        logger = ConsoleLogger()
    else:
        logger = FileLogger(args.logger)

    if state is None:
        manager = SearchManager(space, ds, ds, supernet, logger)
    else:
        manager = SearchManager.from_dict(
            state, FullSearchSpace, ds, ds, supernet, logger
        )

    return manager, path_results


def _train(args: argparse.Namespace) -> None:
    manager, path = _common(args, True, None)

    if args.save_every > 0:
        train_ctx = SaveTrainCtx(
            path / "supernet" / "intermediate", args.save_every
        )
    else:
        train_ctx = None

    manager.train(
        args.epochs,
        build_dl_config(args, args.batch_size),
        args.models_per_batch,
        warm_up=False,
        ctx=train_ctx,
        device=torch.device(args.device)
    )
    _save_manager(path / "supernet", manager)


def _evo(args: argparse.Namespace) -> None:
    with open(Path(args.manager) / "state.json", "r") as f:
        state = json.load(f)

    with open(Path(args.manager) / "weights.pth", "rb") as f:
        state["weights"] = torch.load(f, weights_only=True)

    manager, path = _common(args, False, state)

    if args.save_every > 0:
        evo_ctx = SaveEvolutionCtx(
            path / "evolution" / "intermediate", args.save_every
        )
    else:
        evo_ctx = None

    start_shim_runner()

    try:
        max_flash = _parse_memory(args.flash)
        max_sram = _parse_memory(args.sram)
        fitness = _fitness_wrapper(
            manager._valid_ds.classes, args.resolution, max_flash, max_sram
        )

        top = manager.evolution(
            build_dl_config(args, args.batch_size),
            build_dl_config(args, args.calib_batch_size),
            args.topk,
            args.iterations,
            args.population,
            mutator=mutator,
            fitness=fitness,
            ctx=evo_ctx,
            device=torch.device(args.device)
        )
    finally:
        stop_shim_runner()

    _save_top(path / "evolution", args.save_topk, top)


def _train_sub_parser(sub: argparse._SubParsersAction) -> None:
    train: argparse.ArgumentParser = sub.add_parser(
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
        "--models-per-batch",
        help="the number of models sampled per batch of training data",
        type=int,
        default=1
    )
    train.add_argument(
        "results",
        help="path to store the results of the training",
        type=str
    )
    train.set_defaults(func=_train)


def _evo_sub_parser(sub: argparse._SubParsersAction) -> None:
    evo: argparse.ArgumentParser = sub.add_parser(
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
        "--save-every",
        help="save the intermediate state of the evolution every n iteration",
        type=int,
        default=0
    )
    evo.add_argument(
        "manager",
        help="the path to the manager",
        type=str
    )
    evo.add_argument(
        "results",
        help="path to store the results of the evolution search",
        type=str
    )
    evo.set_defaults(func=_evo)


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    args_argument_logger(parser)
    args_argument_device(parser)

    parser.add_argument(
        "--width-mult",
        required=True,
        help="width multiplier",
        type=float
    )

    args_group_ds(parser)

    sub = parser.add_subparsers()
    _train_sub_parser(sub)
    _evo_sub_parser(sub)

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
