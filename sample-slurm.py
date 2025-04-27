from typing_extensions import Doc
from typing import Annotated, Final
from collections.abc import Iterator


_width_step: Final[float] = 0.1
WIDTH_CHOICES: Annotated[
    Final[list[float]],
    Doc(
        """
        The selected choices for the width multiplier from MCUNet.
        """
    )
] = [0.2 + i * _width_step for i in range(9)]

_resolution_step: Final[int] = 16
RESOLUTION_CHOICES: Annotated[
    Final[list[int]],
    Doc(
        """
        The selected choices for the resolution from MCUNet.
        """
    )
] = [48 + i * _resolution_step for i in range(12)]


def configurations() -> Iterator[tuple[float, int]]:
    """
    Creates all 108 combinations of alpha (width multiplier) and rho (resolution)
    as per the MCUNet paper.

    Returns:
        Iterator[tuple[float, int]]:
            An iterator of all combinations.
    """
    for with_mult in WIDTH_CHOICES:
        for resolution in RESOLUTION_CHOICES:
            yield (with_mult, resolution)


def main() -> None:
    for width_mult, resolution in configurations():
        wm = f"{width_mult}".replace(".", "_")
        test = \
f"""#!/bin/bash

#SBATCH -o ./sample/{wm}-{resolution}.out
#SBATCH -J mcuet-sample-{wm}-{resolution}
#SBATCH --ntasks=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=standard

. ./.venv/bin/activate
python sample-mult.py --width-mult {width_mult} --resolution {resolution} --classes 1000 full /scratch/hclassen/samples/full-{wm}-{resolution}-1000.csv
"""
        with open(f"./sample/slurm/{wm}-{resolution}-1000.slurm", "w") as f:
            f.write(test)

if __name__ == "__main__":
    main()