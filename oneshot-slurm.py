import argparse


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["ilsvrc"]
    )
    parser.add_argument(
        "resolution",
        type=int
    )
    parser.add_argument(
        "widthmult",
        type=float
    )
    parser.add_argument(
        "instance",
        type=str,
        choices=["full", "conv2d", "dwconv2d", "bdwrconv2d"]
    )
    parser.add_argument(
        "cpus",
        type=int
    )
    parser.add_argument(
        "name",
        type=str
    )
    parser.add_argument(
        "traintime",
        type=str
    )
    parser.add_argument(
        "evotime",
        type=str
    )
    parser.add_argument(
        "flash",
        type=str
    )
    parser.add_argument(
        "sram",
        type=str
    )

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    train = f"""#!/bin/bash

#SBATCH -o ./oneshot/{args.name}-{args.dataset}-train.out
#SBATCH -J oneshot-{args.name}-{args.dataset}-train
#SBATCH --ntasks={args.cpus}
#SBATCH --time={args.traintime}
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

module load nvidia/cuda/12.2

. ./.venv/bin/activate
python nas-{args.instance}.py -d cuda:0 -l ./log/oneshot/{args.name}-{args.dataset}-train.log --width-mult {args.widthmult} --resolution {args.resolution} --ds-json --shuffle --num-workers {args.cpus} --pin-memory --prefetch-factor 2 {args.dataset} /scratch/hclassen/subds/ilsvrc-100-train.json train --epochs 120 --models-per-batch 1 --save-every 40 /scratch/hclassen/oneshot/{args.dataset}/{args.name}
"""

    with open(f"oneshot-{args.name}-train.sbatch", "w") as f:
        f.write(train)

    evo = f"""#!/bin/bash

#SBATCH -o ./oneshot/{args.name}-{args.dataset}-evo.out
#SBATCH -J oneshot-{args.name}-{args.dataset}-evo
#SBATCH --ntasks={args.cpus}
#SBATCH --time={args.evotime}
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

module load nvidia/cuda/12.2

. ./.venv/bin/activate
python nas-{args.instance}.py -d cuda:0 -l ./log/oneshot/{args.name}-{args.dataset}-evo.log --width-mult {args.widthmult} --resolution {args.resolution} --ds-json --shuffle --num-workers {args.cpus} --pin-memory --prefetch-factor 2 {args.dataset} /scratch/hclassen/subds/ilsvrc-100-valid.json evo --flash {args.flash} --sram {args.sram} --topk 10 --population 50 --iterations 20 --save-every 5 /scratch/hclassen/oneshot/{args.dataset}/{args.name}/supernet /scratch/hclassen/oneshot/{args.dataset}/{args.name}
"""

    with open(f"oneshot-{args.name}-evo.sbatch", "w") as f:
        f.write(evo)


if __name__ == "__main__":
    main()