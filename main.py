import sys
import json

import torch

import tensorflow as tf

import keras

from mcunet.tinynas import SearchManager
from mcunet.tinynas.searchspace import Model
from mcunet.tinynas.oneshot import SuperNet
from mcunet.tinynas.oneshot.customize import ChoiceLayerMaker, LastChoiceLayerMaker
from mcunet.tinynas.datasets import gtsrb
from mcunet.tinynas.utils import torchhelper

from mcunet.shim import build_model, to_tflite, get_train_datasets

from mcunet.tinyengine.TfliteConvertor import TfliteConvertor
from mcunet.tinyengine.GeneralMemoryScheduler import GeneralMemoryScheduler

from searchspaces import mnasnetplus
from searchspaces.mnasnetplus import simple, more, full


WIDTH_MULT = 0.5
RESOLUTION = 144

GTSRB_LABELS = None
GTSRB_IMAGES = None

FLASH = 1_000_000
SRAM = 320_000

train_ds, _ = get_train_datasets(GTSRB_IMAGES, RESOLUTION, 0.2, 4, True)


def fitness(model: Model) -> bool:
    net = build_model(model, 43)

    # Weirdly converting to tflite format fails without this. Keep this as simple
    # as possible since we only care about converting and not accuracy.
    net.compile(
        optimizer="sgd",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    net.fit(
        tf.ones([1, RESOLUTION, RESOLUTION, 3]), tf.ones([1]), epochs=1, verbose=0
    )

    lite = to_tflite(net, train_ds)

    converter = TfliteConvertor(lite)
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

    return flash <= FLASH and sram <= SRAM


def _simple_manager(ds: gtsrb.CustomDataset) -> SearchManager:
    conv2d_maker = ChoiceLayerMaker(
        mnasnetplus.Conv2dChoicesMaker(), simple.Conv2dSharer
    )
    dwconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.DWConv2dChoicesMaker(), simple.DWConv2dSharer
    )
    bdwrconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.BDWRConv2dChoicesMaker(), simple.BDWRConv2dSharer
    )

    blocks_maker = mnasnetplus.ChoiceBlocksMaker(
        conv2d_maker, dwconv2d_maker, bdwrconv2d_maker
    )

    last_maker = LastChoiceLayerMaker(mnasnetplus.LastLayerSharer)

    warm_up_setter = simple.SimpleWarmUpSetter()

    supernet = SuperNet(
        blocks_maker,
        last_maker,
        warm_up_setter,
        ds.classes,
        WIDTH_MULT,
        initialize_weights=False
    )

    manager = SearchManager(
        mnasnetplus.MnasNetPlus(WIDTH_MULT, RESOLUTION, False),
        ds,
        supernet
    )

    return manager


def _more_manager(ds: gtsrb.CustomDataset) -> SearchManager:
    conv2d_maker = ChoiceLayerMaker(
        mnasnetplus.Conv2dChoicesMaker(), more.Conv2dSharer
    )
    dwconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.DWConv2dChoicesMaker(), more.DWConv2dSharer
    )
    bdwrconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.BDWRConv2dChoicesMaker(), more.BDWRConv2dSharer
    )

    blocks_maker = mnasnetplus.ChoiceBlocksMaker(
        conv2d_maker, dwconv2d_maker, bdwrconv2d_maker
    )

    last_maker = LastChoiceLayerMaker(mnasnetplus.LastLayerSharer)

    warm_up_setter = more.MoreWarmUpSetter()

    supernet = SuperNet(
        blocks_maker,
        last_maker,
        warm_up_setter,
        ds.classes,
        WIDTH_MULT,
        initialize_weights=False
    )

    manager = SearchManager(
        mnasnetplus.MnasNetPlus(WIDTH_MULT, RESOLUTION, True),
        ds,
        supernet
    )

    return manager


def _full_manager(ds: gtsrb.CustomDataset) -> SearchManager:
    conv2d_maker = ChoiceLayerMaker(
        mnasnetplus.Conv2dChoicesMaker(), full.Conv2dSharer
    )
    dwconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.DWConv2dChoicesMaker(), full.DWConv2dSharer
    )
    bdwrconv2d_maker = ChoiceLayerMaker(
        mnasnetplus.BDWRConv2dChoicesMaker(), full.BDWRConv2dSharer
    )

    blocks_maker = mnasnetplus.ChoiceBlocksMaker(
        conv2d_maker, dwconv2d_maker, bdwrconv2d_maker
    )

    last_maker = LastChoiceLayerMaker(mnasnetplus.LastLayerSharer)

    warm_up_setter = full.FullWarmUpSetter()

    supernet = SuperNet(
        blocks_maker,
        last_maker,
        warm_up_setter,
        ds.classes,
        WIDTH_MULT,
        initialize_weights=False
    )

    manager = SearchManager(
        mnasnetplus.MnasNetPlus(WIDTH_MULT, RESOLUTION, True),
        ds,
        supernet
    )

    return manager


def _run(
    manager: SearchManager, ds: gtsrb.CustomDataset, results: str, device
) -> None:
    manager = _simple_manager(ds)
    manager.train(epochs=100, warm_up_epochs=25, device=device)

    saved = manager.to_dict()
    weights = saved["weights"]
    del saved["weights"]

    with open(f"{results}/manager.json", "w") as f:
        json.dump(saved, f)

    with open(f"{results}/weights.pth", "wb") as f:
        torch.save(weights, f)

    top = manager.evolution(fitness=fitness, device=device)
    for i, (candidate, accuracy) in enumerate(top):
        with open(f"{results}/top-{i}-{accuracy:.5f}".replace(".", "_"), "wb") as f:
            torch.save(candidate.state_dict(), f)


def main(results1: str, results2: str, results3: str) -> None:
    ds = gtsrb.CustomDataset(
        GTSRB_LABELS,
        GTSRB_IMAGES,
        transform=torchhelper.transform_resize(RESOLUTION)
    )

    device = torchhelper.get_device()
    _run(_simple_manager(ds), ds, results1, device)
    _run(_more_manager(ds), ds, results2, device)
    _run(_full_manager(ds), ds, results3, device)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise RuntimeError("missing arguments")

    GTSRB_LABELS = f"{sys.argv[1]}/training/labels.csv"
    GTSRB_IMAGES = f"{sys.argv[1]}/training/images/"

    main(*sys.argv[2:5])
