from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class IdentityOp(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        _: torch.Tensor,
        forward: Callable[[torch.Tensor], torch.Tensor],
        backward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        # If a MixedOp is the first layer of a NN, ``tensor`` is the overall
        # input to this net. This input tensor doesn't require grad. However
        # to be able to use ``torch.autograd.grad`` in the backward pass, every
        # input to it must require grad and must be part of the autograd DAG.
        # This if statement handles this specific case.
        if not tensor.requires_grad:
            tensor = tensor.clone()
            tensor.requires_grad = True

        with torch.enable_grad():
            output = forward(tensor)

        ctx.save_for_backward(tensor, output)
        ctx.backward = backward

        return output.detach()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        tensor, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, tensor, grad)

        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward(tensor, output, grad)

        return grad_x[0], binary_grads, None, None


class MixedOpSwitch():
    active: bool

    def __init__(self) -> None:
        self.active = False


class MixedOp(nn.Module):
    n_choices = int
    candidates: list[nn.Module]
    arch_params: nn.Parameter  # AP_path_alpha / architecture parameters
    binary_gates: nn.Parameter  # AP_path_wb / binary gates

    active_index: int | None
    inactive_index: int | None

    log_prob: float | None
    current_prob_over_ops: float | None

    _switch: MixedOpSwitch

    def __init__(
        self, candidates: list[nn.Module], switch: MixedOpSwitch
    ) -> None:
        super().__init__()

        self.n_choices = len(candidates)

        self.candidates = nn.ModuleList(candidates)
        self.arch_params = nn.Parameter(torch.Tensor(self.n_choices))
        self.binary_gates = nn.Parameter(torch.Tensor(self.n_choices))

        self.active_index = 0
        self.inactive_index = list(range(1, self.n_choices))

        self.log_prob = None
        self.current_prob_over_ops = None

        self._switch = switch

    def softmax_arch_params(self) -> torch.Tensor:
        return F.softmax(self.arch_params, dim=0)

    def chosen_index(self) -> int:
        softmax = self.softmax_arch_params()
        return int(torch.argmax(softmax))

    def chosen_candidate(self) -> nn.Module:
        return self.candidates[self.chosen_index()]

    def random_candidate(self) -> None:
        return np.random.choice(self.candidates)

    def entropy(self, eps: float = 1e-8) -> torch.Tensor:
        softmax = self.softmax_arch_params()
        log = torch.log(softmax + eps)
        return -torch.sum(torch.mul(softmax, log))

    def set_chosen_candidate(self) -> None:
        idx = self.chosen_index()
        self.active_index = idx
        self.inactive_index = \
            list(range(0, idx)) + list(range(idx + 1, self.n_choices))

    def _forward_wrapper(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def forward(tensor: torch.Tensor) -> torch.Tensor:
            return self.candidates[self.active_index](tensor)

        return forward

    def _backward_wrapper(
        self
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        def backward(
            tensor: torch.Tensor, output: torch.Tensor, grads: torch.Tensor
        ) -> torch.Tensor:
            binary_grads = torch.zeros_like(self.binary_gates)
            with torch.no_grad():
                for k in range(len(self.candidates)):
                    if k != self.active_index:
                        out_k = self.candidates[k](tensor)
                    else:
                        out_k = output

                    binary_grads[k] = torch.sum(out_k * grads)

            return binary_grads

        return backward

    # implements only MixedEdge.MODE == 'full_v2'
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._switch.active:
            return self.candidates[self.active_index](tensor)

        return ArchGradientFunction.apply(
            tensor,
            self.binary_gates,
            self._forward_wrapper(),
            self._backward_wrapper()
        )

    # implements only MixedEdge.MODE == 'full_v2'
    def binarize(self) -> None:
        """
        prepare: active_index, inactive_index, AP_path_wb, log_prob (optional),
        current_prob_over_ops (optional)
        """
        self.log_prob = None

        # reset binary gates
        self.binary_gates.data.zero_()

        # binarize according to probs
        probs = self.softmax_arch_params()

        sample = torch.multinomial(probs, 1)[0].item()
        self.active_index = sample
        self.inactive_index = \
            list(range(0, sample)) + list(range(sample + 1, self.n_choices))

        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs

        # set binary gate
        self.binary_gates.data[sample] = 1.0

        # avoid over-regularization
        for i in range(self.n_choices):
            for _, param in self.candidates[i].named_parameters():
                param.grad = None

    # implements only MixedEdge.MODE == 'full_v2'
    def set_arch_param_grad(self) -> None:
        binary_grads = self.binary_gates.grad.data

        if self.arch_params.grad is None:
            self.arch_params.grad = torch.zeros_like(self.arch_params.data)

        probs = self.softmax_arch_params()
        for i in range(self.n_choices):
            for j in range(self.n_choices):
                # delta_ij is defined as:
                #            + - 1, i == j
                # delta_ij = +
                #            + - 0, else
                self.arch_params.grad.data[i] += \
                    binary_grads[j] * probs[j] * (int(i == j) - probs[i])

    # no rescale_updated_arch_param() as it's only used with MixedEdge.MODE == 'two'
