import sys
import json
import subprocess
from pathlib import Path

from ..tinynas.searchspace import  Model


__all__ = ["start_shim_runner", "stop_shim_runner", "memory_footprint"]


def _create_subprocess(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )


def _destroy_subprocess(sub: subprocess.Popen) -> None:
    msg = json.dumps("end")
    sub.stdin.write(f"{msg}\n".encode("utf-8"))
    sub.stdin.flush()

    try:
        sub.wait(1)
    except subprocess.TimeoutExpired:
        sub.terminate()


class SubprocessManager():
    _path: Path
    _n: int
    _children: list[subprocess.Popen] | None

    _limit: int
    _count: int

    def __init__(
        self,
        path: str | Path,
        n: int,
        limit: int = 1000,
        lazy: bool = False
    ) -> None:
        if isinstance(path, str):
            path = Path(str)
        self._path = path
        self._n = n

        self._children = None
        if not lazy:
            self._create_children()

        self._limit = limit
        self._count = 0

    def __del__(self) -> None:
        self._destroy_children()

    def _create_children(self) -> None:
        if self._children is not None:
            raise RuntimeError("children were already created")

        self._children = [
            _create_subprocess(self._path) for _ in range(self._n)
        ]

    def _destroy_children(self) -> None:
        if self._children is None:
            return

        for child in self._children:
            _destroy_subprocess(child)

        self._children = None

    def submit(self, msg: str) -> list[str]:
        if self._children is None:
            self._create_children()

        if self._count == self._limit:
            first = self._children.pop(0)
            _destroy_subprocess(first)

            self._count = 0
            self._children.append(_create_subprocess(self._path))

        first = self._children[0]
        first.stdin.write(f"{msg}\n".encode("utf-8"))
        first.stdin.flush()

        self._count += 1

        results = []
        while True:
            line = first.stdout.readline().decode("utf-8").rstrip()

            if line == "end":
                break

            results.append(line)

        return results

    def stop(self) -> None:
        self._destroy_children()


_shim_runner: SubprocessManager | None = None


def start_shim_runner() -> None:
    global _shim_runner

    if _shim_runner is not None:
        return

    path = Path(__file__).parent / "_sub.py"
    _shim_runner = SubprocessManager(path, 2, 1000, lazy=False)


def stop_shim_runner() -> None:
    global _shim_runner

    if _shim_runner is None:
        return

    _shim_runner.stop()
    _shim_runner = None


def memory_footprint(
    model: Model, classes: int, resolution: int
) -> tuple[int, int]:
    if _shim_runner is None:
        raise RuntimeError("shim runner must be started first")

    results = _shim_runner.submit(
        json.dumps({
            "model": model.to_dict(),
            "classes": classes,
            "resolution": resolution
        })
    )

    data = json.loads(results[-1])
    return data["flash"], data["sram"]
