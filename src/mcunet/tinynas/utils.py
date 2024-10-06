__all__ = ["make_divisible", "make_caption"]


def make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """
    Change ``v`` so that it is divisble by ``divisor``. Copied from
    https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py.

    Args:
        v (float):
            The value.
        divisor (int):
            The newe divisor for ``v``.
        min_value (int, None):
            Lower bound for ``v`` to not go below.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_caption(text: str, line_length: int, divider: str) -> str:
    diff = line_length - len(text)
    dashes = int((diff - 2) / 2) * divider

    caption = f"{dashes} {text} {dashes}"
    return f"{caption}{divider * (line_length - len(caption))}"
