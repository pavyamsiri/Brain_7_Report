# Standard modules
import os

import yaml

from data_stream import StreamType

# Hide pygame import message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import multiprocessing as mp
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

# External modules
import numpy as np
import pandas as pd
import pygame
from scipy import optimize

# Internal modules
from aliases import FilePath
from models import Event, ModelBase, ModifiedModel
from spikerbox import Control, SpikerBox


CLASSIFICATION_FILE: FilePath = "recordings/Actual Event Times.csv"


count = 0


class DataSample(NamedTuple):
    """Struct that contains data for one file sample."""

    file_name: FilePath
    events: List[Tuple[Event, float]]


class CounterArgs(NamedTuple):
    stream_file: FilePath
    model: ModelBase
    buffer_time: float
    update_factor: float
    wait_time: float
    num_samples: int
    quality_factor: float
    filter_flag: bool
    events: List[Tuple[Control, float]]


def main():
    # Load SpikerBox parameters
    CONFIG_PATH: FilePath = "settings/spiker_box.srconfig"
    with open(CONFIG_PATH, "r") as config_file:
        config_data: Dict = yaml.safe_load(config_file)
        buffer_time: float = float(config_data["buffer_time"])
        update_factor: float = float(config_data["update_factor"])
        wait_time: float = float(config_data["wait_time"])
        quality_factor: float = float(config_data["quality_factor"])
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

    data: List[DataSample] = load_data_samples(CLASSIFICATION_FILE)

    # SpikerBox parameters
    spb_parameters: List[float] = [
        buffer_time,
        update_factor,
        wait_time,
        num_samples,
        quality_factor,
    ]

    # Modified simple classifier model parameters
    model_parameters: List[float] = [
        m_event_threshold,
        positive_amplitude,
        negative_amplitude,
        spacing,
    ]
    model: ModelBase = ModifiedModel(*model_parameters)
    minimisation_function = lambda x: 1 - calculate_streaming_accuracy(
        data, ModifiedModel(*x[0:4]), True, *x[4:]
    )
    start_time: float = time.time()
    optimisation_flag: bool = False
    if optimisation_flag:
        # Takes around 400 function evaluations which can take on the order of 10 hours to complete.
        res = optimize.minimize(
            minimisation_function,
            (model_parameters + spb_parameters),
        )
        model_parameters = res.x[0:4]
        spb_parameters = res.x[4:]
    model: ModelBase = ModifiedModel(*model_parameters)
    accuracy = 100 * (calculate_streaming_accuracy(data, model, True, *spb_parameters))
    print(f"Took {time.time() - start_time:.2f} seconds")
    print(f"The accuracy is {accuracy:.2f}%")
    print(f"The SpikerBox parameters are\n{spb_parameters}")
    print(f"The model parameters are\n{model_parameters}")


def load_data_samples(classification_path: FilePath) -> List[DataSample]:
    df: pd.DataFrame = pd.read_csv(classification_path)
    data: List[DataSample] = []
    for idx in df.index.values:
        file_root: str = df.loc[idx, "File"]
        file_name: FilePath = f"../SpikerStreamPython/Recordings/{file_root}.npy"
        events_str: List[str] = list(df.loc[idx, "Actual events"])
        events: List[Control] = []
        for event in events_str:
            if event == "L":
                events.append(Control.LEFT)
            elif event == "R":
                events.append(Control.RIGHT)
        times_str: List[str] = df.loc[idx, "Times"].split(", ")
        times: List[float] = [float(timestamp) for timestamp in times_str]
        data.append(DataSample(file_name, list(zip(events, times))))
    return data


def count_predictions(args: CounterArgs) -> Tuple[float, int]:
    s: SpikerBox = SpikerBox(
        args.model,
        args.buffer_time,
        args.update_factor,
        args.wait_time,
        args.num_samples,
        args.quality_factor,
        StreamType.ARR,
        args.stream_file,
        cport=None,
        filter_flag=args.filter_flag,
        plot_flag=False,
    )

    correct_left: int = 0
    predicted_left: int = 0
    correct_right: int = 0
    predicted_right: int = 0

    events: List[Tuple[Control, float]] = args.events
    clock: pygame.time.Clock = pygame.time.Clock()
    controls: Dict[Control, int] = {
        Control.LEFT: 0,
        Control.RIGHT: 0,
    }
    labels: List[Tuple[Control, float]] = []
    restart_flag: bool = False
    while not restart_flag:
        tick: int = clock.tick()
        controls: Dict[Control, int] = s.process_input(controls, tick)
        restart_flag = s.is_restarting()
        timestamp: Optional[float] = s.timestamps[-1] if len(s.timestamps) > 0 else None
        moved: Optional[Control] = None
        for key in controls:
            if controls[key] == 1:
                labels.append((key, timestamp))
                controls[key] = 0
                moved = key
        if moved is not None and timestamp is not None:
            time_difference: List[float] = []
            for _, actual_timestamp in events:
                time_difference.append(np.abs(timestamp - actual_timestamp))
            # If there any possible timestamps left
            if len(time_difference) > 0:
                min_idx = np.argmin(time_difference)
                if events[min_idx][0] == moved:
                    correct_left += 1 if events[min_idx][0] == Control.LEFT else 0
                    correct_right += 1 if events[min_idx][0] == Control.RIGHT else 0
                events.pop(min_idx)
        if moved is not None:
            predicted_left += 1 if moved == Control.LEFT else 0
            predicted_right += 1 if moved == Control.RIGHT else 0

    # Total predictions
    total_predicted: int = predicted_left + predicted_right

    # Accuracy
    total_accuracy: float = 0
    # Left accuracy
    if predicted_left > 0:
        total_accuracy += correct_left / predicted_left
    # Right accuracy
    if predicted_right > 0:
        total_accuracy += correct_right / predicted_right
    # Average accuracy
    if predicted_left > 0 and predicted_right > 0:
        total_accuracy /= 2

    return total_accuracy, total_predicted


def calculate_streaming_accuracy(
    data_samples: List[DataSample],
    model: ModelBase,
    filter_flag: bool,
    buffer_time: float,
    update_factor: float,
    wait_time: float,
    num_samples: int,
    quality_factor: float,
) -> float:
    global count
    print(f"{count=}")
    count += 1
    function_arguments = []
    for data_sample in data_samples:
        args = CounterArgs(
            stream_file=data_sample.file_name,
            model=model,
            buffer_time=buffer_time,
            update_factor=update_factor,
            wait_time=wait_time,
            num_samples=int(num_samples),
            quality_factor=quality_factor,
            filter_flag=filter_flag,
            events=data_sample.events,
        )
        function_arguments.append(args)
    results = []
    with mp.Pool(processes=len(data_samples)) as p:
        results = p.map(count_predictions, function_arguments)

    divisor: int = 0
    total_accuracy: float = 0

    for acc, total_predictions in results:
        total_accuracy += total_predictions * acc
        divisor += total_predictions

    if divisor > 0:
        total_accuracy /= divisor
    return total_accuracy


if __name__ == "__main__":
    main()
