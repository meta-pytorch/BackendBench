# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import threading
import time
from dataclasses import asdict, dataclass, fields

import matplotlib.pyplot as plt
from pynvml import (
    NVML_CLOCK_ID_CURRENT,
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
    NVML_FI_DEV_POWER_CURRENT_LIMIT,
    NVML_FI_DEV_POWER_INSTANT,
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetClock,
    nvmlDeviceGetFieldValues,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTemperature,
    nvmlInit,
    nvmlShutdown,
)

# query every 10 ms
DEFAULT_QUERY_INTERVAL = 0.01


@dataclass
class PowerEvent:
    timestamp: float
    sm_clock: float
    mem_clock: float
    power_draw_instant: float
    power_draw_current_limit: float
    gpu_temp: float


def check_nvml_status(nvml_status):
    if nvml_status:
        raise RuntimeError("NVML initialization failed")


class GPUCollectorThread:
    def __init__(self, gpu_id=None, query_interval=DEFAULT_QUERY_INTERVAL) -> None:
        self.gpu_id = int(gpu_id) if gpu_id else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        # Assume Python GIL so not protecting this using Atomics
        self.continue_monitoring = True
        # Sampling interval in seconds
        self.sampling_interval = query_interval
        self.events = []
        self.iter = []
        check_nvml_status(nvmlInit())

    def start(self):
        handle = nvmlDeviceGetHandleByIndex(int(self.gpu_id))
        while self.continue_monitoring:
            # check gpu power event
            sm_clock = nvmlDeviceGetClock(handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
            mem_clock = nvmlDeviceGetClock(handle, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT)
            power_info = nvmlDeviceGetFieldValues(
                handle, [NVML_FI_DEV_POWER_INSTANT, NVML_FI_DEV_POWER_CURRENT_LIMIT]
            )
            gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            self.events.append(
                PowerEvent(
                    timestamp=int(time.time_ns() / 1e3),
                    sm_clock=sm_clock,
                    mem_clock=mem_clock,
                    power_draw_instant=power_info[0].value.uiVal / 1000.0,
                    power_draw_current_limit=power_info[1].value.uiVal / 1000.0,
                    gpu_temp=gpu_temp,
                )
            )
            time.sleep(self.sampling_interval)
        nvmlShutdown()


class PowerManager:
    def __init__(self) -> None:
        self.gpu_id = None
        self.output_dir = None
        self.query_interval = None

    def start(self) -> None:
        self.collector = GPUCollectorThread(self.gpu_id, self.query_interval)
        self._t = threading.Thread(target=self.collector.start)
        self._t.start()

    def stop(self) -> None:
        self.collector.continue_monitoring = False
        self._t.join()

    def finalize(self) -> None:
        # flush results to file
        result_file = os.path.join(self.output_dir, "power.csv")
        with open(result_file, "w", newline="") as csvfile:
            # Get the field names from the dataclass to use as CSV header
            fieldnames = [field.name for field in fields(PowerEvent)]

            # Create a DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")

            # Write the header row
            writer.writeheader()

            total_energy = 0
            # Write each dataclass instance as a row in the CSV
            current_interval = self.query_interval
            for i in range(len(self.collector.events)):
                if i < len(self.collector.events) - 1:
                    current_interval = (
                        self.collector.events[i + 1].timestamp - self.collector.events[i].timestamp
                    ) / 1e6
                event = self.collector.events[i]
                total_energy += event.power_draw_instant * current_interval
                writer.writerow(asdict(event))
        return total_energy


def plot_power_charts(benchmark_name: str, gpu_id: int, output_dir: str, power_csv_file: str):
    # Read CSV
    with open(power_csv_file) as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)  # first row as header
        header = [col.strip() for col in header]
        data = {col: [] for col in header}

        for row in reader:
            for col, value in zip(header, row):
                value = float(value)
                data[col].append(value)

    # Generate synthetic time axis (100 ms per sample)
    n_samples = len(next(iter(data.values())))
    time = [
        (data["timestamp"][i] - data["timestamp"][0]) / 1000.0 for i in range(n_samples)
    ]  # seconds (0.1s = 100 ms)

    # Plot power chart
    plt.figure(figsize=(10, 6))
    for power_col in header[3:5]:
        plt.plot(time, data[power_col], label=power_col)
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.title(f"{benchmark_name} power consumption over time on device {gpu_id}")
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-power.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # Plot temp chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[5:]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature (C)")
    plt.legend()
    plt.title(f"{benchmark_name} temperature over time on device {gpu_id}")
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-temp.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # Plot frequency chart
    plt.figure(figsize=(10, 6))
    for temp_col in header[1:3]:
        plt.plot(time, data[temp_col], label=temp_col)
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (MHz)")
    plt.legend()
    plt.title(f"{benchmark_name} frequency over time on device {gpu_id}")
    plt.savefig(
        os.path.join(output_dir, f"{benchmark_name}-freq.png"),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    plot_power_charts("addmm", 0, "./bench_power", "./bench_power/power.csv")
