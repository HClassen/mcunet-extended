import time
from typing import cast
from copy import deepcopy
from itertools import islice
from abc import abstractmethod, ABC
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.ops import Conv2dNormActivation

from .. import SearchSpace
from ..mobilenet import (
    FIRST_CONV_CHANNELS,
    LAST_CONV_CHANNELS,
    DROPOUT,
    LAYER_SETTINGS,
    build_first,
    build_pool,
    build_classifier,
    skeletonnet_train,
    skeletonnet_valid,
    MobileSkeletonNet
)
from ..searchspace import Layer, Block, Model
from ..searchspace.layers import module_to_layer
from ..searchspace.model import build_model
from ..utils import make_divisible, make_caption, Logger

from .layers import ChoiceBlockConstructor, ChoiceBlock, LastChoiceLayer
from .share import ParameterSharerConstructor


"""
Single Path One-Shot Neural Architecture Search

This subproject provides a base class for the super-network. Subclasses need to
provide several methods to agment the behavior of the super-network for their
needs. Additionally, several helper methods are provided as well as the needed
`crossover` and `mutate` functions for the evolution search.

The `BaseSuperNet` and evolution functions are modeled after the paper "Single
Path One-Shot Neural Architecture Search with Uniform Sampling".

It can be found: https://arxiv.org/abs/1904.00420
"""


__all__ = [
    "OneShotNet",
    "SuperNet", "WarmUpCtx", "supernet_warm_up", "supernet_train",
    "initial_population", "crossover", "mutation", "evaluate"
]


class OneShotNet(MobileSkeletonNet):
    _width_mult: float
    _block_lengths: list[int]

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        classifier: nn.Sequential,
        width_mult: float,
        block_lengths: list[int]
    ) -> None:
        super().__init__(first, blocks, last, pool, classifier, False)

        self._block_lengths = block_lengths
        self._width_mult = width_mult

    def to_model(self) -> Model:
        """
        Build a `Model` from this `OneShotNet`.

        Returns:
            Model:
                The configuration of this network.
        """

        accum = 0
        blocks: list[Block] = []
        for length in self._block_lengths:
            modules = self.blocks[accum:accum + length]
            blocks.append(
                Block([module_to_layer(module) for module in modules])
            )

            accum += length

        return Model(self._width_mult, blocks, self.last[0].out_channels)


class SuperNet(MobileSkeletonNet):
    first: Conv2dNormActivation
    blocks: list[ChoiceBlock]
    last: LastChoiceLayer

    _width_mult: float

    def __init__(
        self,
        block_constructor: ChoiceBlockConstructor,
        last_sharer: ParameterSharerConstructor,
        classes: int,
        width_mult: float,
        dropout: float = DROPOUT,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
        initialize_weights: bool = True
    ) -> None:
        round_nearest = 8
        in_channels = (
            make_divisible(FIRST_CONV_CHANNELS * width_mult, round_nearest),
        )

        # Build the first conv2d layer.
        first = build_first(in_channels[0], norm_layer, activation_layer)

        # Build the 7 `ChoiceBlocks`.
        blocks: list[ChoiceBlock] = []
        for setting in LAYER_SETTINGS:
            block, out_channels = block_constructor(
                in_channels, setting[1], setting[3], width_mult, round_nearest,
                norm_layer, activation_layer
            )

            blocks.append(block)

            in_channels = out_channels

        # Build the last conv2d layer.
        last_out = make_divisible(
            LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
        )

        last = LastChoiceLayer(
            last_sharer, in_channels, last_out, norm_layer, activation_layer
        )

        pool = build_pool()
        classifier = build_classifier(classes, last_out, dropout)

        self._width_mult = width_mult

        super().__init__(
            first, blocks, last, pool, classifier, initialize_weights
        )

    def _weight_initialization(self) -> None:
        nn.init.kaiming_normal_(self.first[0].weight, mode="fan_out")
        nn.init.ones_(self.first[1].weight)
        nn.init.zeros_(self.first[1].bias)

        for block in self.blocks:
            block._weight_initialization()

        self.last.sharer._weight_initialization()

        nn.init.normal_(self.classifier[1].weight, 0, 0.01)
        nn.init.zeros_(self.classifier[1].bias)

    def set(self, model: Model) -> None:
        """
        Sets the super-network to a sample from the search space.

        Args:
            model (Model):
                The sampled model from the search space.

        Raises:
            Exception:
                If `model` does not match the choices available in the
                super-network.
        """
        if len(self.blocks) != len(model.blocks):
            raise Exception(f"invalid blocks length {len(model.blocks)}")

        for choice_block, model_block in zip(self.blocks, model.blocks):
            choice_block = cast(ChoiceBlock, choice_block)

            choice_block.set(model_block)

        self.last.set(model.blocks[-1].layers[-1].out_channels)

    def unset(self) -> None:
        """
        Unsets the active sample of the search space this super-network has been
        set to.
        """
        for block in self.blocks:
            block.unset()

        self.last.unset()

    def get(self, copy: bool = False) -> OneShotNet:
        """
        Collects the current active operations in a single net. A call to
        `set` is requiered beforehand.

        Args:
            copy (bool):
                If `True` perform a deepcopy on all operations specific to the
                current active operations.

        Returns:
            OneShotNet:
                The concrete network.
        """
        blocks: list[nn.Module] = []
        block_lengths: list[int] = []

        for block in self.blocks:
            block = cast(ChoiceBlock, block)

            layers = block.get(copy)
            block_lengths.append(len(layers))
            blocks.extend(layers)

        last = self.last.get(copy)

        return OneShotNet(
            self.first if not copy else deepcopy(self.first),
            blocks,
            last,
            self.pool,
            self.classifier if not copy else deepcopy(self.classifier),
            self._width_mult,
            block_lengths
        )


class WarmUpCtx(ABC):
    @abstractmethod
    def set(
        self, supernet: SuperNet, max_model: Model, max_net: MobileSkeletonNet
    ) -> None:
        """
        Called to make use of the trained weights in a maximum model.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
            max_model (Model):
                The configuration of the current maximum model of the search
                space.
            max_net (MobileSkeletonNet):
                The actual Pytorch network of `max_model` with the trained
                weights.
        """
        pass

    def pre(self, supernet: SuperNet, device=None) -> None:
        """
        Called before the first maximum model is trained. This is optional.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
            device:
                The Pytorch device the network will be moved to.
        """
        pass

    def post(self, supernet: SuperNet, device=None) -> None:
        """
        Called after the last maximum model is trained. This is optional.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
            device:
                The Pytorch device the network was moved to.
        """
        pass


def supernet_warm_up(
    supernet: SuperNet,
    space: SearchSpace,
    ctx: WarmUpCtx,
    ds: Dataset,
    epochs: int,
    batch_size: int,
    *,
    logger: Logger,
    batches: int | None = None,
    device=None
) -> None:
    """
    Warm up a `SuperNet` by training the maximum model(s) from the search
    space. Set the weights of the super-netowrk to the trained weights from
    the maximum model(s).

    Args:
        supernet (SuperNet):
            The super-network to warm up.
        space (SearchSpace):
            The search space to get the maximum models.
        ctx (WarmUpCtx):
            The warm up context.
        ds (Dataset):
            The data set to train on.
        epochs (int):
            The number of training epochs.
        batch_size (int):
            The number of samples per batch.
        logger (Logger):
            The interface to pass logging information to.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    warm_up_start = time.time()

    ctx.pre(supernet, device=device)

    logger.log(make_caption("Warm Up", 70, " "))
    for i, max_model in enumerate(space.max_models()):
        net = build_model(
            max_model, supernet.classifier[-1].weight.shape[0]
        )

        skeletonnet_train(
            net, ds, epochs, batch_size, 0.9, 5e-5,
            logger=logger, batches=batches, device=device
        )

        logger.log(make_caption(f"Set Parameters({i + 1})", 70, "-"))

        set_start = time.time()
        ctx.set(supernet, max_model, net)
        set_time = time.time() - set_start

        logger.log(f"time={set_time:.2f}s\n")

    ctx.post(supernet)

    warm_up_time = time.time() - warm_up_start
    logger.log(f"\ntotal={warm_up_time:.2f}s\n")


class TrainCtx(ABC):
    @abstractmethod
    def epoch(self, epoch: int, supernet: SuperNet, epochs: int) -> None:
        """
        Gets called at the end of every epoch during training.

        Args:
            epoch (int):
                The current epoch.
            supernet (SuperNet):
                The super-network as is after a training epoch.
            epochs (int):
                The total number of epochs.
        """
        pass


def supernet_train(
    supernet: SuperNet,
    space: SearchSpace,
    ds: Dataset,
    initial_lr: float,
    epochs: int,
    batch_size: int,
    models_per_batch: int,
    *,
    logger: Logger,
    ctx: TrainCtx | None = None,
    batches: int | None = None,
    device=None
) -> None:
    """
    Train a `SuperNet` by sampling random models per batch of training data,
    setting the super-network to the sampled model and performing the backward
    pass.

    Args:
        supernet (SuperNet):
            The super-network to train.
        space (SearchSpace):
            The search space from which to sample the models.
        ds (Dataset):
            The data set to train on.
        initial_lr (float):
            The initial learning rate for cosine annealing learning.
        epochs (int):
            The number of training epochs.
        batch_size (int):
            The number of samples per batch.
        models_per_batch (int):
            The number of models sampled per batch.
        logger (Logger):
            The interface to pass logging information to.
        ctx (TrainCtx, None):
            The train context.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    train_start = time.time()

    models = iter(space)
    dl = DataLoader(ds, batch_size, shuffle=True)
    batches = batches if batches is not None else len(dl)

    supernet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(supernet.parameters(), lr=initial_lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 256
    )

    logger.log(make_caption("Training", 70, " "))
    supernet.train()
    for i in range(epochs):
        logger.log(make_caption(f"Epoch {i + 1}/{epochs}", 70, "-"))

        epoch_start = time.time()

        for k, (images, labels) in enumerate(dl):
            lr = scheduler.get_last_lr()[0]
            logger.log(
                f"epoch={i + 1}, batch={k + 1:0{len(str(batches))}}/{batches}, lr={lr:.05f}",
                end=""
            )

            batch_start = time.time()

            images = images.to(device)
            labels = labels.to(device)

            for _ in range(models_per_batch):
                model = next(models)
                supernet.set(model)

                outputs = supernet(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            scheduler.step()

            batch_time = time.time() - batch_start

            logger.log(f", time={batch_time:.2f}s")

            if batches == k + 1:
                break

        supernet.unset()
        if ctx is not None:
            ctx.epoch(i + 1, supernet, epochs)

        epoch_time = time.time() - epoch_start
        logger.log(f"\ntime={epoch_time:.2f}s\n")

    train_time = time.time() - train_start
    logger.log(f"\ntotal={train_time:.2f}s\n")


def initial_population(
    space: Iterator[Model],
    size: int,
    fitness: Callable[[Model], bool]
) -> list[Model]:
    """
    Select the initial population of models for evolution search.

    Args:
        space (Iterator[Model]):
            An iterator over the search space, which (randomly) samples models.
        size (int):
            The initial population size.
        fitness (Callable[[Model], bool]):
            A function to evaluate the fitness of a model for some criterion
            but not accuracy.

    Returns:
        list[Model]:
            The selected population.
    """
    population: list[OneShotNet] = []

    for _ in range(size):
        model = next(space)

        while not fitness(model):
            model = next(space)

        population.append(model)

    return population

def _combine(width_mult: float, top: list[Block], bottom: list[Block]) -> Model:
    layer = bottom[0].layers[0]
    in_channels = top[-1].layers[-1].out_channels
    bottom[0].layers[0] = Layer(
        layer.op,
        in_channels,
        layer.out_channels,
        layer.kernel_size,
        layer.stride,
        layer.expansion_ratio,
        layer.shortcut,
        layer.se_ratio
    )

    blocks = top + bottom
    return Model(width_mult, blocks, blocks[-1].layers[-1].out_channels)


def crossover(p1: Model, p2: Model) -> tuple[Model, Model]:
    """
    Performs single point crossover on `p1` and `p1`. Randomly select a split
    point `s` of the seven inner blocks. Then select the blocks `0` to `s` from
    `p1` and merge them with the blocks `s + 1` to `6` from `p2` to create the
    first offspring model. The second offspring is produces by flipping `p1` and
    `p2`. The split point `s` is chosen between [1, min_block_length - 1].

    At the merge point for both models, set the `in_channels` of the bottom
    half to the `out_channels` of the top half.

    Args:
        p1 (Model):
            The first parent.
        p2 (Model):
            The second parent.

    Returns:
        tuple[Model, Model]:
            The two offsprings.
    """
    l = min(len(p1.blocks), len(p2.blocks))
    split = torch.randint(1, l, [1]).item()

    p1_top = deepcopy(p1.blocks[:split])
    p1_bottom = deepcopy(p1.blocks[split:])

    p2_top = deepcopy(p2.blocks[:split])
    p2_bottom = deepcopy(p2.blocks[split:])

    o1 = _combine(p1.width_mult, p1_top, p2_bottom)
    o2 = _combine(p2.width_mult, p2_top, p1_bottom)

    return o1, o2


def mutation(
    model: Model, p: float, mutator: Callable[[int, float, Block], Block]
) -> Model:
    """
    Mutates the structure of `Model`. Each block of the model is passed to the
    supplied `mutator` with a propability of `p`.

    Args:
        model (Model):
            The model to mutate.
        p (float):
            The mutation probability.
        mutator (Callable[[int, float, Block], None]):
            Mutates a `Block` and returns it. It receives the index of the block,
            the width multiplicator of the model and the block itself.

    Returns:
        Model:
            The new and mutated model.
    """
    mutated = deepcopy(model)
    blocks: list[Block] = []
    for i, block in enumerate(model.blocks):
        copied = deepcopy(block)

        if torch.rand(1).item() < p:
            copied = mutator(i, mutated.width_mult, copied)

        blocks.append(copied)

    for i in range(1, len(blocks)):
        prev = blocks[i - 1]
        this = blocks[i]

        if prev.layers[-1].out_channels == this.layers[0].in_channels:
            continue

        layer = this.layers[0]
        in_channels = prev.layers[-1].out_channels
        this.layers[0] = Layer(
            layer.op,
            in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.expansion_ratio,
            layer.shortcut,
            layer.se_ratio
        )

    return Model(model.width_mult, blocks, blocks[-1].layers[-1].out_channels)


def evaluate(
    population: list[Model],
    supernet: SuperNet,
    valid_ds: Dataset,
    valid_batch_size: int,
    calib_ds: Dataset,
    calib_batch_size: int,
    *,
    valid_batches: int | None = None,
    calib_batches: int = 20,
    device=None
) -> list[float]:
    """
    Evaluates the candidates of a population regarding their accuracy. Each
    model inherits its weight from the `supernet`. Before evaluating a model,
    the running mean and var of its batch-normalization recalibrated using
    the `calib_ds`.

    Args:
        population (list[Model]):
            The population of one iteration in evolution search.
        supernet (SuperNet):
            A (pre-trained) supernet to inherit weights from.
        valid_ds (Dataset):
            The data set used to evaluate the accuracy of a model.
        valid_batch_size (int):
            The number of samples per validation batch.
        calib_ds (Dataset):
            The data set used to recalibrate the batch-normalizations of a model.
        calib_batch_size (int):
            The number of samples per calibration batch.
        valid_batches (int, None):
            The number of batches per epoch used during validation. If set to
            `None` use the whole data set.
        calib_batches (int):
            The number of batches used for recalibration.
        device:
            The Pytorch device the network is moved to.

    Returns:
        list[float]:
            The accuracies of the model in `population` in the same order.
    """
    results: list[float] = []

    calib_dl = DataLoader(calib_ds, calib_batch_size, shuffle=True)

    for candidate in population:
        supernet.set(candidate)
        net = supernet.get(True)

        net.to(device)

        # Recalibrate batch norm statistics (running mean/var).
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        net.train()
        for (images, _) in islice(calib_dl, calib_batches):
            images = images.to(device)
            net(images)

        # Insert based on descending accuracy.
        accuracy = skeletonnet_valid(
            net, valid_ds, valid_batch_size, batches=valid_batches, device=device
        )
        results.append(accuracy)

    supernet.unset()
    return results
