from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def paths(workspace: str | Path) -> tuple[Path, Path]:
    root = Path(workspace) / ".agent"
    root.mkdir(parents=True, exist_ok=True)
    return root / "daemon.pid", root / "daemon.log"


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def health(port: int, timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=timeout
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 健康检查失败就是没起来
        return None


def supervise(config_path: str, port: int, workspace: str, interval: float = 2.0) -> int:
    """守护循环：子进程退出即按退避重启，收到 SIGTERM 后优雅收尾。"""
    pid_file, log_file = paths(workspace)
    pid_file.write_text(str(os.getpid()))
    stopping = {"flag": False}

    def _stop(signum, frame):  # noqa: ARG001
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    delay = interval
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n=== supervisor {os.getpid()} start {time.strftime('%F %T')} ===\n")
        while not stopping["flag"]:
            started = time.time()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "doubao_agent.cli",
                    "--config",
                    config_path,
                    "serve",
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=log,
            )
            log.write(f"[supervisor] child {child.pid} started\n")
            log.flush()

            while child.poll() is None:
                if stopping["flag"]:
                    break
                time.sleep(0.5)

            if stopping["flag"]:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break

            uptime = time.time() - started
            delay = interval if uptime > 60 else min(delay * 2, 30)
            log.write(
                f"[supervisor] child exit={child.returncode} uptime={uptime:.0f}s "
                f"restart_in={delay:.0f}s\n"
            )
            log.flush()
            for _ in range(int(delay * 2)):
                if stopping["flag"]:
                    break
                time.sleep(0.5)

    pid_file.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doubao-agent.daemon")
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workspace", default="/workspace")
    args = parser.parse_args(argv)
    return supervise(args.config, args.port, args.workspace)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
