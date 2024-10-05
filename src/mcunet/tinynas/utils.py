__all__ = ["make_caption"]


def make_caption(text: str, line_length: int, divider: str) -> str:
    diff = line_length - len(text)
    dashes = int((diff - 2) / 2) * divider

    caption = f"{dashes} {text} {dashes}"
    return f"{caption}{divider * (line_length - len(caption))}"
