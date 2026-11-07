#!/usr/bin/env python3
"""Log GPU temperature/power/clocks every 10s; detect a disappearing GPU."""
from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "gpu_monitor.csv"
OUT.parent.mkdir(exist_ok=True)
FIELDS = ["timestamp", "temp_c", "power_w", "sm_mhz", "mem_mhz", "util_pct", "mem_used_mib"]


def main() -> None:
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            w.writeheader()
        while True:
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"], timeout=8, stderr=subprocess.DEVNULL).decode().strip()
                parts = [x.strip() for x in out.split(",")]
                row = dict(zip(["temp_c", "power_w", "sm_mhz", "mem_mhz", "util_pct", "mem_used_mib"], parts))
            except Exception as e:
                row = {"temp_c": "MISSING", "power_w": "", "sm_mhz": "", "mem_mhz": "",
                       "util_pct": "", "mem_used_mib": ""}
            row["timestamp"] = ts
            w.writerow(row)
            f.flush()
            time.sleep(10)


if __name__ == "__main__":
    main()
