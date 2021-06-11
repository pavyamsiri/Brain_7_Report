# Standard modules
import glob
from operator import pos
import os
import time
from typing import Dict, List, Tuple

# External modules
import numpy as np
from numpy.core.numeric import count_nonzero
from scipy import signal
from scipy import optimize
import yaml
from aliases import FilePath
from models import Event, ModelBase, ModifiedModel

# Internal modules
from utils import *


SNIPPETS_FOLDER: FilePath = "snippets"
count: int = 0


def main():
    # Load SpikerBox parameters
    CONFIG_PATH: FilePath = "settings/spiker_box.srconfig"
    with open(CONFIG_PATH, "r") as config_file:
        config_data: Dict = yaml.safe_load(config_file)
        num_samples: int = int(config_data["num_samples"])
        # Modified simple classifier parameters
        m_event_threshold: int = int(
            config_data["classifier"]["MSC"]["event_threshold"]
        )
        positive_amplitude: float = float(
            config_data["classifier"]["MSC"]["positive_amplitude"]
        )
        negative_amplitude: float = float(
            config_data["classifier"]["MSC"]["negative_amplitude"]
        )
        spacing: float = float(config_data["classifier"]["MSC"]["spacing"])
    model_parameters: List[float] = [
        m_event_threshold,
        positive_amplitude,
        negative_amplitude,
        spacing,
    ]
    file_paths: List[FilePath] = glob.glob(f"{SNIPPETS_FOLDER}/*.npy")
    start_time = time.time()
    optimisation_flag: bool = False
    if optimisation_flag:
        # Snippet accuracy optimisation does not take too long
        minimisation_function = lambda x: 1 - calculate_snippet_accuracy(
            ModifiedModel(*x[:-1]), file_paths, x[-1]
        )
        res = optimize.minimize(
            minimisation_function,
            model_parameters + [num_samples],
            method="nelder-mead",
        )
        model_parameters = res.x[:-1]
        num_samples: int = int(res.x[-1])
    model: ModelBase = ModifiedModel(*model_parameters)
    accuracy: float = 100 * calculate_snippet_accuracy(model, file_paths, num_samples)
    print(f"Took {time.time() - start_time:.2f} seconds")
    print(f"The accuracy is {accuracy:.2f}%")
    print(f"The model parameters are\n{model_parameters}")
    print(f"The number of samples is {num_samples}")


def calculate_snippet_accuracy(
    model: ModelBase, file_paths: List[FilePath], num_samples: int
) -> float:
    global count
    print(f"{count=}")
    count += 1
    correct_left: int = 0
    predicted_left: int = 0
    correct_right: int = 0
    predicted_right: int = 0
    correct_noise: int = 0
    predicted_noise: int = 0

    for file_path in file_paths:
        event = get_snippet_event(file_path)
        if event != "left" and event != "right" and event != "noise":
            continue
        else:
            # Load snippet
            signal_slice, time_slice = parse_snippet(np.load(file_path))
            # Standardise and downsample signal
            processed_data = signal.resample(signal_slice, int(num_samples))
            label = model.classify(processed_data)
            if str(label) == event:
                correct_left += 1 if label == Event.LEFT else 0
                correct_right += 1 if label == Event.RIGHT else 0
                correct_noise += 1 if label == Event.NOISE else 0
            predicted_left += 1 if label == Event.LEFT else 0
            predicted_right += 1 if label == Event.RIGHT else 0
            predicted_noise += 1 if label == Event.NOISE else 0

    # Accuracy
    total_accuracy: float = 0
    # Number of unique labels
    divisors: int = 0
    # Left accuracy
    if predicted_left > 0:
        total_accuracy += correct_left / predicted_left
        divisors += 1
    # Right accuracy
    if predicted_right > 0:
        total_accuracy += correct_right / predicted_right
        divisors += 1
    # Noise accuracy
    if predicted_noise > 0:
        total_accuracy += correct_noise / predicted_noise
        divisors += 1
    # Average accuracy
    if divisors > 0:
        total_accuracy /= divisors

    return total_accuracy


if __name__ == "__main__":
    main()
