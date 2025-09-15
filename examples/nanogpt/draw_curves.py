# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import matplotlib.pyplot as plt


def read_log_file(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at: {log_path}")

    with open(log_path, "r") as file:
        log_lines = file.readlines()

    return log_lines


def parse_training_logs(log_lines):
    results = {}

    # Regular expression to match lines with training metrics
    # Original pattern
    pattern = r"iter (\d+): loss ([\d\.]+), time ([\d\.]+)ms, backendbench overhead time ([\d\.]+)ms, time exclude backendbench overhead ([\d\.]+), mfu ([\d\.\-]+)%"

    for line in log_lines:
        # Try to match with the new pattern first
        match = re.search(pattern, line)
        if match:
            iteration = int(match.group(1))
            if iteration % 10 != 0:
                continue
            loss = float(match.group(2))
            time_ms = float(match.group(3))
            overhead_time_ms = float(match.group(4))
            time_exclude_ms = float(match.group(5))
            mfu_percent = float(match.group(6))

            results[iteration] = {
                "loss": loss,
                "time": time_ms,
                "overhead_time": overhead_time_ms,
                "time_exclude": time_exclude_ms,
                "mfu": mfu_percent,
            }

    return results


def draw_forward_time_curves(
    results_list,
    labels=None,
    title="Forward Time per Iteration",
    save_path=None,
    figsize=(10, 6),
    dpi=100,
    x_label="Iteration",
    y_label="Forward Time (ms)",
):
    """
    Draw forward time curves from multiple results dictionaries.
    """
    # Create figure and axis
    plt.figure(figsize=figsize, dpi=dpi)

    # Define a list of colors for distinguishing multiple curves
    colors = [
        "blue",
        "red",
    ]

    # Plot each results dictionary as a separate line
    for i, results in enumerate(results_list):
        # Extract iterations and forward time values
        iterations = list(results.keys())
        forward_time_values = [results[it]["time"] for it in iterations]

        # Sort by iteration to ensure the line is drawn correctly
        sorted_data = sorted(zip(iterations, forward_time_values))
        iterations = [item[0] for item in sorted_data]
        forward_time_values = [item[1] for item in sorted_data]

        # Plot the line
        plt.plot(
            iterations,
            forward_time_values,
            linestyle="-",
            color=colors[i],
            label=labels[i],
            marker="",
            linewidth=1,
        )

    # Add labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    # Tighten layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)

    return plt.gcf()  # Return the figure object


def draw_loss_curves(
    results_list,
    labels=None,
    title="Training Loss vs Iteration",
    save_path=None,
    figsize=(10, 6),
    dpi=100,
    x_label="Iteration",
    y_label="Loss",
):
    """
    Draw loss curves from multiple results dictionaries.
    """
    # Create figure and axis
    plt.figure(figsize=figsize, dpi=dpi)

    # Define a list of colors for distinguishing multiple curves
    colors = [
        "blue",
        "red",
    ]

    # Plot each results dictionary as a separate line
    for i, results in enumerate(results_list):
        # Extract iterations and loss values
        iterations = list(results.keys())
        loss_values = [results[it]["loss"] for it in iterations]

        # Sort by iteration to ensure the line is drawn correctly
        sorted_data = sorted(zip(iterations, loss_values))
        iterations = [item[0] for item in sorted_data]
        loss_values = [item[1] for item in sorted_data]

        # Plot the line
        plt.plot(
            iterations,
            loss_values,
            linestyle="-",
            color=colors[i],
            label=labels[i],
            marker="",
            linewidth=1,
        )

    # Add labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    # Tighten layout
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)

    return plt.gcf()  # Return the figure object


def process_and_plot_logs(log_paths, labels=None, save_path_prefix=None, first_n_lines=None):
    """
    Utility function to process multiple log files and plot both loss and forward time curves.
    """
    results_list = []

    # Process each log file
    for log_path in log_paths:
        log_lines = read_log_file(log_path)
        results = parse_training_logs(log_lines)
        results_list.append(results)

    # Determine save paths for the figures
    loss_save_path = f"{save_path_prefix}_loss.png"
    forward_time_save_path = f"{save_path_prefix}_time.png"

    # Draw loss curves
    loss_fig = draw_loss_curves(results_list, labels=labels, save_path=loss_save_path)

    # Draw forward time curves
    forward_time_fig = draw_forward_time_curves(
        results_list, labels=labels, save_path=forward_time_save_path
    )

    return loss_fig, forward_time_fig, results_list


# Example usage:
if __name__ == "__main__":
    # Example for multiple log files
    os.makedirs("figs", exist_ok=True)
    log_paths = ["logs/log.txt", "logs/backendbench_log.txt"]
    labels = ["PyTorch Aten", "LLM generated"]
    loss_fig, forward_time_fig, all_results = process_and_plot_logs(
        log_paths, labels, save_path_prefix="figs/training_comparison"
    )

    # create figs if not exist
    os.makedirs("figs", exist_ok=True)

    # Save the figures under figs foler
    plt.show()
