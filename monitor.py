import time
import psutil


def main() -> None:
    with open("monitor.log", "w") as f:
        while True:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            used = memory.used / 1024**3
            free = memory.available / 1024**3
            available = memory.available * 100 / memory.total

            now = time.localtime()
            formatted = time.strftime("%Y-%m-%d %H:%M:%S%z", now)
            f.write(f"[{formatted}] cpu={cpu}%, memory-used=({used:.2f}GiB, {memory.percent:.2f}%), memory-available=({free:.2f}GiB, {available:.2f}%)\n")
            f.flush()

            time.sleep(60.0 * 10)


if __name__ == "__main__":
    main()
