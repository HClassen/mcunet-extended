import torch

from mcunet.tinynas import SampleManager
from mcunet.tinynas.searchspace import configurations
from mcunet.tinynas.searchspace.model import build_model
from mcunet.tinynas.configurations.mnasnetplus import MnasNetPlus

import mcunet.shim as shim

from mcunet.tinyengine.TfliteConvertor import TfliteConvertor
from mcunet.tinyengine.GeneralMemoryScheduler import GeneralMemoryScheduler

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


def measure(model, width_mult, resolution) -> tuple[int, int, int]:
    net1 = build_model(model, 1000)
    flops = net1.flops(resolution, device=torch.device("cuda:0"))

    net2 = shim.build_model(model, 1000)
    shim.dummy_train(net2, resolution)
    tflite = shim.dummy_to_tflite(net2, resolution)

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

    del net1
    del net2
    del tflite
    shim.clear_keras()

    return flops, flash, sram


def main() -> None:
    spaces = [MnasNetPlus(a, b) for a, b in configurations()]
    manager = SampleManager(spaces)

    manager.sample(1000)

    for space, results in manager.apply(measure):
        name = space.__class__.__name__.lower()
        width_mult = str(space.width_mult)[:3].replace(".", "_")

        with open(f"{name}-{width_mult}-{space.resolution}.csv", "w") as f:
            f.write("flops,flash,sram\n")

            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]}\n")


if __name__ == "__main__":
    main()
