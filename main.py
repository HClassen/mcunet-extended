import json

import torch

import tensorflow as tf

import keras

from mcunet.tinynas import SearchManager
from mcunet.tinynas.mnasnet import Model
from mcunet.tinynas.datasets import gtsrb
from mcunet.tinynas.utils import torchhelper

from mcunet.shim import build_model, to_tflite, get_train_datasets

from mcunet.tinyengine.TfliteConvertor import TfliteConvertor
from mcunet.tinyengine.GeneralMemoryScheduler import GeneralMemoryScheduler


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


def main() -> None:
    ds = gtsrb.CustomDataset(
        GTSRB_LABELS,
        GTSRB_IMAGES,
        transform=torchhelper.transform_resize(RESOLUTION)
    )

    device = torchhelper.get_device()

    manager = SearchManager((WIDTH_MULT, RESOLUTION), ds)
    manager.train(device=device)

    saved = manager.to_dict()
    weights = saved["weights"]
    del saved["weights"]

    with open("./manager.json", "w") as f:
        json.dump(saved, f)

    with open("./weights.pth", "wb") as f:
        torch.save(weights, f)

    top = manager.evolution(fitness=fitness, device=device)
    for i, (candidate, accuracy) in enumerate(top):
        with open(f"./top-{i}-{accuracy:.5f}".replace(".", "_"), "wb") as f:
            torch.save(candidate.state_dict(), f)

if __name__ == "__main__":
    if GTSRB_LABELS is None:
        raise Exception("path to GTSRB labels is missing")

    if GTSRB_IMAGES is None:
        raise Exception("path to GTSRB images is missing")

    main()
