import sys
import json
from pathlib import Path
from shutil import rmtree
from collections.abc import Callable

import torch

from mcunet.tinynas import SearchManager
from mcunet.tinynas.oneshot import SuperNet, OneShotNet
from mcunet.tinynas.searchspace import Model
from mcunet.tinynas.datasets import transform_resize
from mcunet.tinynas.datasets import gtsrb
from mcunet.tinynas.configurations.mnasnetplus import (
    super_choice_blocks_wrapper,
    LastLayerSharer,
    MnasNetPlus
)
from mcunet.tinynas.configurations.mnasnetplus.share import simple

import mcunet.shim as shim

from mcunet.tinyengine.TfliteConvertor import TfliteConvertor
from mcunet.tinyengine.GeneralMemoryScheduler import GeneralMemoryScheduler

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


WIDTH_MULT = 0.5
RESOLUTION = 144

FLASH = 1_000_000
SRAM = 320_000


def _save_manager(path: Path, manager: SearchManager) -> None:
    if path.exists():
        rmtree(path)
        path.unlink()
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


def _fitness_wrapper(classes: int, resolution: int) -> Callable[[Model], bool]:
    def fitness(model: Model) -> bool:
        net = shim.build_model(model, classes)
        shim.dummy_train(net, resolution)
        tflite = shim.dummy_to_tflite(net, resolution)

        converter = TfliteConvertor(tflite)
        converter.parseOperatorInfo()

        out = []
        layer = converter.layer
        memory_scheduler = GeneralMemoryScheduler(
            layer,
            False,
            False,
            outputTables=out,
            inplace=True,
            mem_visual_path=None,
            VisaulizeTrainable=False,
        )
        memory_scheduler.USE_INPLACE = True
        memory_scheduler.allocateMemory()

        flash = memory_scheduler.flash
        sram = memory_scheduler.buffers["input_output"]

        del net
        del tflite
        shim.clear_keras()

        return flash <= FLASH and sram <= SRAM

    return fitness


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("path to data set or save results is missing")

    path_results = Path(sys.argv[2])
    if not path_results.exists():
        path_results.mkdir()

    path_ds = Path(sys.argv[1])
    gtsrb.get(path_ds)
    ds = gtsrb.GTSRBDataset(
        path_ds / "training" / "labels.csv",
        path_ds / "training" / "images",
        transform=transform_resize(RESOLUTION)
    )

    space = MnasNetPlus(WIDTH_MULT, RESOLUTION)

    block_constructor = super_choice_blocks_wrapper(
        simple.Conv2dSharer, simple.DWConv2dSharer, simple.BDWRConv2dSharer
    )
    last_sharer = LastLayerSharer
    supernet = SuperNet(
        block_constructor,
        last_sharer,
        ds.classes,
        WIDTH_MULT,
        initialize_weights=False
    )

    manager = SearchManager(space, ds, supernet)
    manager.train(
        epochs=300,
        batch_size=256,
        models_per_batch=4,
        warm_up=True,
        warm_up_ctx=simple.SimpleWarmUpCtx(),
        warm_up_epochs=25,
        warm_up_batch_size=256,
        batches=1,
        device=torch.device("cuda:0")
    )
    _save_manager(path_results / "e300", manager)

    fitness = _fitness_wrapper(ds.classes, RESOLUTION)
    top = manager.evolution(fitness=fitness, device=torch.device("cuda:0"))
    _save_top(path_results / "e300", 5, top)

    supernet = SuperNet(
        block_constructor,
        last_sharer,
        ds.classes,
        WIDTH_MULT,
        initialize_weights=False
    )

    manager = SearchManager(space, ds, supernet)
    manager.train(
        epochs=500,
        batch_size=256,
        models_per_batch=4,
        warm_up=True,
        warm_up_ctx=simple.SimpleWarmUpCtx(),
        warm_up_epochs=25,
        warm_up_batch_size=256,
        batches=1,
        device=torch.device("cuda:0")
    )
    _save_manager(path_results / "e500", manager)

    top = manager.evolution(fitness=fitness, device=torch.device("cuda:0"))
    _save_top(path_results / "e500", 5, top)


if __name__ == "__main__":
    main()
