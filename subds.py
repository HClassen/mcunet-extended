import json
import argparse
from pathlib import Path

from mcunet.tinynas.datasets import imagenet, gtsrb

from helper import *


def _gtsrb_split(args: argparse.Namespace) -> None:
    ds = ds_load("gtsrb", args.path, "train", args.ds_json, None)

    split = gtsrb.strict_split(ds, args.n)
    if len(split) == len(args.n) + 1:
        args.n.append(-1)

    path = Path(args.results)
    for subds, size in zip(split, args.n):
        with open(path / f"n{size}.json", "w") as f:
            json.dump(subds.to_dict(), f)


def _gtsrb_sub_parser(sub: argparse._SubParsersAction) -> None:
    gtsrb_parser: argparse.ArgumentParser = sub.add_parser(
        "gtsrb",
        help="transform functions for the GTSRB data set"
    )
    gtsrb_parser.add_argument(
        "path",
        help="the path to the data set",
        type=str
    )
    gtsrb_parser.add_argument(
        "results",
        help="the path to the save the results",
        type=str
    )

    gtsrb_sub: argparse._SubParsersAction = gtsrb_parser.add_subparsers()
    gtsrb_split: argparse.ArgumentParser = gtsrb_sub.add_parser(
        "split",
        help="split the GTSRB data set"
    )
    gtsrb_split.add_argument(
        "n",
        help="the path to the data set",
        type=int,
        nargs="+"
    )
    gtsrb_split.set_defaults(func=_gtsrb_split)


def _ilsvrc_split(args: argparse.Namespace) -> None:
    ds = ds_load("ilsvrc", args.path, "train", args.ds_json, None)

    split = imagenet.strict_split(ds, args.n)
    if len(split) == len(args.n) + 1:
        args.n.append(-1)

    path = Path(args.results)
    for subds, n in zip(split, args.n):
        with open(path / f"n{n}.json", "w") as f:
            json.dump(subds.to_dict(), f)


def _ilsvrc_subset(args: argparse.Namespace) -> None:
    ds = ds_load("ilsvrc", args.path, "train", args.ds_json, None)

    path = Path(args.results)
    for classes in args.n:
        subds = imagenet.subset(ds, int(classes), None)

        with open(path / f"c{classes}.json", "w") as f:
            json.dump(subds.to_dict(), f)


def _ilsvrc_sub_parser(sub: argparse._SubParsersAction) -> None:
    ilsvrc_parser: argparse.ArgumentParser = sub.add_parser(
        "ilsvrc",
        help="transform functions for the ILSVRC data set"
    )
    ilsvrc_parser.add_argument(
        "path",
        help="the path to the data set",
        type=str
    )
    ilsvrc_parser.add_argument(
        "results",
        help="the path to the save the results",
        type=str
    )

    ilsvrc_sub: argparse._SubParsersAction = ilsvrc_parser.add_subparsers()

    ilsvrc_split: argparse.ArgumentParser = ilsvrc_sub.add_parser(
        "split",
        help="split the ILSVRC data set"
    )
    ilsvrc_split.add_argument(
        "n",
        help="the path to the data set",
        type=int,
        nargs="+"
    )
    ilsvrc_split.set_defaults(func=_ilsvrc_split)

    ilsvrc_subset: argparse.ArgumentParser = ilsvrc_sub.add_parser(
        "subset",
        help="create a subset the ILSVRC data set"
    )
    ilsvrc_subset.add_argument(
        "n",
        help="the path to the data set",
        type=int,
        nargs="+"
    )
    ilsvrc_subset.set_defaults(func=_ilsvrc_subset)


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds-json",
        help="load the data set from a JSON file instead of a folder",
        action="store_true"
    )

    sub = parser.add_subparsers()
    _gtsrb_sub_parser(sub)
    _ilsvrc_sub_parser(sub)

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    if "func" not in args:
        parser.print_help()
        return

    path = Path(args.results)
    path.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()