import time
from copy import deepcopy
from typing import cast, Any
from abc import ABC, abstractmethod
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
    MobileSkeletonNet
)
from ..searchspace import Block, Model
from ..searchspace.layers import module_to_layer, BaseModule
from ..searchspace.model import build_model
from ..utils import make_divisible, make_caption

from .helper import reduce_channels
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
    "initial_population", "crossover", "mutate"
]


class OneShotNet(MobileSkeletonNet):
    _block_lengths: list[int]

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        classifier: nn.Sequential,
        block_lengths: list[int]
    ) -> None:
        super().__init__(first, blocks, last, pool, classifier, False)

        self._block_lengths = block_lengths

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

        return Model(blocks, self.last[0].out_channels)


class SuperNet(MobileSkeletonNet):
    first: Conv2dNormActivation
    blocks: list[ChoiceBlock]
    last: LastChoiceLayer

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
        Set the super-network to a sample from the search space.

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
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    warm_up_start = time.time()

    ctx.pre(supernet, device=device)

    print(make_caption("Warm Up", 70, " "))
    for i, max_model in enumerate(space.max_models()):
        net = build_model(
            max_model, supernet.classifier[-1].weight.shape[0]
        )

        skeletonnet_train(
            net, ds, epochs, batch_size, 0.9, 5e-5,
            batches=batches, device=device
        )

        print(make_caption(f"Set Parameters({i + 1})", 70, "-"))

        set_start = time.time()
        ctx.set(supernet, max_model, net)
        set_time = time.time() - set_start

        print(f"time={set_time:.2f}s\n")

    ctx.post(supernet)

    warm_up_time = time.time() - warm_up_start
    print(f"\ntotal={warm_up_time:.2f}s\n")


def supernet_train(
    supernet: SuperNet,
    space: SearchSpace,
    ds: Dataset,
    initial_lr: float,
    epochs: int,
    batch_size: int,
    models_per_batch: int,
    *,
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(supernet.parameters(), lr=initial_lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 256
    )

    parallel = nn.DataParallel(supernet)
    supernet.to(parallel.src_device_obj)

    print(make_caption("Training", 70, " "))
    supernet.train()
    for i in range(epochs):
        print(make_caption(f"Epoch {i + 1}/{epochs}", 70, "-"))

        epoch_start = time.time()

        for k, (images, labels) in enumerate(dl):
            lr = scheduler.get_last_lr()[0]
            print(f"epoch={i + 1}, batch={k + 1:0{len(str(batches))}}/{batches}, lr={lr:.05f}", end="")

            batch_start = time.time()

            labels = labels.to(parallel.output_device)

            for _ in range(models_per_batch):
                model = next(models)
                supernet.set(model)

                outputs = parallel(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            scheduler.step()

            batch_time = time.time() - batch_start

            print(f", time={batch_time:.2f}s")

            if batches == k + 1:
                break

        epoch_time = time.time() - epoch_start
        print(f"\ntime={epoch_time:.2f}s\n")

    train_time = time.time() - train_start
    print(f"\ntotal={train_time:.2f}s\n")


def initial_population(
    space: Iterator[Model],
    size: int,
    supernet: SuperNet,
    fitness: Callable[[Model], bool]
) -> list[OneShotNet]:
    """
    Select the initial population of models for evolution search.

    Args:
        space (Iterator[Model]):
            An iterator over the search space, which (randomly) samples models.
        size (int):
            The initial population size.
        supernet (SuperNet):
            A (pre trained) instance of `SuperNet` to get the actual NNs from.
        fitness (Callable[[Model], bool]):
            A function to evaluate the fitness of a model for some criterion.

    Returns:
        list[Model]:
            The selected population.
    """
    population: list[OneShotNet] = []

    for _ in range(size):
        model = next(space)

        while not fitness(model):
            model = next(space)

        supernet.set(model)
        population.append(supernet.get(True))

    return population


def crossover(p1: OneShotNet, p2: OneShotNet) -> OneShotNet:
    """
    Performs single point crossover on `p1` and `p1`. Randomly select a split
    point `s` of the seven inner blocks. Then select the blocks `0` to `s` from
    `p1` and merge them with the blocks `s + 1` to `6` from `p2` to create the
    offspring model. `s` is chosen from [0, 5], so that at least one block of a
    parent is part of the child.

    To adjust for different `out_channels` of the last selected layer from `p1`
    and `in_channels` of the first selected layer from `p2` take the result of
    `min(out_channels, in_channels)`. If `out_channels` is the minimum then
    reduce `in_channels` to `out_channels`. If `in_channels` is the minimum
    reduce the `in_channels` and `out_channels` of all layers in the `p1` block
    to the minimum.

    Take the first conv2d layer from `p1`. Take the last conv2d layer and
    classification layer from `p1`.

    Args:
        p1 (OneShotNet):
            The first parent.
        p2 (OneShotNet):
            The second parent.

    Returns:
        OneShotNet:
            The offspring model.
    """
    split = torch.randint(0, 6, [1]).item()

    p1_split_lengths = p1._block_lengths[:split + 1]
    p2_split_lengths = p2._block_lengths[split + 1:]

    p1_split = sum(p1_split_lengths)
    p2_split = len(p2.blocks) - sum(p2_split_lengths)

    p1_blocks = deepcopy(p1.blocks[:p1_split])
    p2_blocks = deepcopy(p2.blocks[p2_split:])

    p1_out_channels = p1_blocks[-1].out_channels
    p2_in_channels = p2_blocks[0].in_channels

    if p1_out_channels < p2_in_channels:
        # Only need to adjust the in_channels of the first p2 layer.
        p2_blocks[0] = reduce_channels(
            p2_blocks[0], p1_out_channels, p2_blocks[0].out_channels
        )
    elif p2_in_channels < p1_out_channels:
        # All layers of the last p1 block need their out_channels adjusted.
        # For all but the first of this block also adjust the in_channels.
        start = sum(p1_split_lengths[:-1])
        p1_blocks[start] = reduce_channels(
            p1_blocks[start],
            p1_blocks[start].in_channels,
            p2_in_channels
        )

        for i in range(start + 1, p1_split):
            p1_blocks[i] = reduce_channels(
                p1_blocks[i],
                p2_in_channels,
                p2_in_channels
            )

    blocks = p1_blocks + p2_blocks

    return OneShotNet(
        deepcopy(p1.first),
        blocks,
        deepcopy(p2.last),
        p2.pool,
        deepcopy(p2.classifier),
        p1_split_lengths + p2_split_lengths
    )


def mutate(oneshot: OneShotNet, p: float) -> OneShotNet:
    """
    Mutate the weights of the conv2d and linear layers. For each weights if the
    threshold `p` is passed add gaussian noise with std = 0.1.

    Args:
        oneshot (OneShotNet):
            The net to mutate.
        p (float):
            The mutation probability.

    Returns:
        OneShotNet:
            The mutated net.
    """
    # Helper to mutate only Linear and Conv2d.
    def maybe_mutate(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)) and torch.rand(1).item() < p:
            with torch.no_grad():
                m.weight += torch.rand_like(m.weight) * 0.1

    first = deepcopy(oneshot.first)
    for m in first.modules():
        maybe_mutate(m)

    blocks: list[BaseModule] = []
    for block in oneshot.blocks:
        block = deepcopy(block)

        for m in block.modules():
            maybe_mutate(m)

        blocks.append(block)

    last = deepcopy(oneshot.last)
    for m in last.modules():
        maybe_mutate(m)

    classifier = deepcopy(oneshot.classifier)
    for m in classifier.modules():
        maybe_mutate(m)

    return OneShotNet(
        first,
        blocks,
        last,
        oneshot.pool,
        classifier,
        oneshot._block_lengths
    )
