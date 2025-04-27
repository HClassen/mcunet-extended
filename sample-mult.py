import argparse
from pathlib import Path
from itertools import islice
from collections.abc import Callable

from mcunet.tinynas.searchspace import Model, SearchSpace
from mcunet.tinynas.searchspace.model import build_model
from mcunet.tinynas.instances import full, conv2d, dwconv2d, bdwrconv2d
from mcunet.shim.runner import (
    memory_footprint,
    start_shim_runner,
    stop_shim_runner
)


_instances: dict[str, Callable[[float, int], SearchSpace]] = {
    "full": full.FullSearchSpace,
    "conv2d": conv2d.Conv2dSearchSpace,
    "dwconv2d": dwconv2d.DWConv2dSearchSpace,
    "bdwrconv2d": bdwrconv2d.BDWRConv2dSearchSpace
}


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width-mult",
        required=True,
        help="width multiplier",
        type=float
    )
    parser.add_argument(
        "--resolution",
        help="resolution of the images",
        required=True,
        type=int
    )
    parser.add_argument(
        "--classes",
        help="classes to learn",
        required=True,
        type=int
    )
    parser.add_argument(
        "space",
        help="select the search space to use",
        type=str,
        choices=["full", "conv2d", "dwconv2d", "bdwrconv2d"]
    )
    parser.add_argument(
        "result",
        help="path to store the result",
        type=str
    )

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    ctor = _instances.get(args.space)
    if ctor is None:
        raise ValueError(f"unknown search space {args.space}")

    space = ctor(args.width_mult, args.resolution)

    path: Path = Path(args.result)
    if path.parent.is_dir():
        path.parent.mkdir(exist_ok=True, parents=True)

    file = open(path, "w")
    file.write("flops,flash,sram\n")

    start_shim_runner()
    try:
        for model in islice(space, 1000):
            flops = build_model(model, args.classes).flops(args.resolution)
            flash, sram = memory_footprint(model, args.classes, args.resolution)

            file.write(f"{flops},{flash},{sram}\n")
    finally:
        stop_shim_runner()
        file.close()


if __name__ == "__main__":
    main()