"""General program function:
    1. Take a directory as input
    2. Pair .txt and .wav files
    3. Loop through pairs
        i. Parse .txt files to get timestamps
        ii. For each timestamp cut out a snippet around it size `snippet_size`.
        iii. Save each snippet as npy array named {original_filename}_{event_name}_{event_number}.npy
        in target folder with the first row being the signal and the second row being its time
"""
import argparse
import glob
import json
import os
import sys
import wave
from typing import List, Tuple

import scipy
import numpy as np
from matplotlib import pyplot as plt
import yaml
from spikerbox import NOISE_FREQUENCY
from train_models import CONFIG_PATH

from utils import normalise_signal


def main(args: List = sys.argv[1:]):
    """Main entry point.

    Parameters
    ----------
    args : List
        unparsed command line arguments. Default value is sys.argv.
    """
    # Parse command line arguments
    parsed_args = _parse_args(args)
    input_directory = parsed_args.input_directory.rstrip("/")
    output_directory = parsed_args.output_directory.rstrip("/")
    default_snippet_size = parsed_args.snippet_size
    default_right_proportion = parsed_args.right_proportion

    # Check if input directory exists
    if not os.path.isdir(input_directory):
        # Exit if does not exist
        print(f"The directory {input_directory} does not exist! Exiting...")
        sys.exit()

    # Check if output directory exists
    if not os.path.isdir(output_directory):
        # If not then check if at least its parent directories exist
        head, _ = os.path.split(output_directory)
        head = "./" if head == "" else head
        if not os.path.isdir(head):
            # Exit if parent directories don't exist
            print(
                f"The parent directories of the output directory {output_directory} does not exist! Exiting..."
            )
            sys.exit()
        # Otherwise make the output directory
        try:
            os.mkdir(output_directory)
        except OSError as e:
            print(f"Failed to create the output directory! {e}")
            sys.exit()

    # Check if plots folder exists if plot_flag is true
    if not os.path.isdir(output_directory + "/Plots") and parsed_args.plot_flag:
        try:
            os.mkdir(output_directory + "/Plots")
        except OSError as e:
            print(f"Failed to create the plots folder! {e}")
            sys.exit()

    # Check if config file was passed in
    config = None
    # Check if config file path exists
    if parsed_args.config is not None and not os.path.isfile(parsed_args.config):
        print("Config file does not exist! Proceeding with standard settings.")
    elif parsed_args.config is not None:
        config = None
        with open(parsed_args.config) as json_file:
            config = json.load(json_file)

    completed = {True: [], False: []}

    # Load SpikerBox args
    with open(CONFIG_PATH, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        num_samples = int(config_data["num_samples"])
        quality_factor = float(config_data["quality_factor"])

    # Create filter
    filter_b, filter_a = scipy.signal.iirnotch(
        NOISE_FREQUENCY, quality_factor, fs=10000
    )

    # wav files
    file_type = "npy" if not parsed_args.wav_flag else "wav"

    # Go through files in input directory
    for name in glob.glob(f"{input_directory}/*.{file_type}"):
        # File path excluding .wav
        root_name = name.rstrip(f".{file_type}")
        # Tail of file path excluding .wav
        _, tail = os.path.split(name)
        tail = tail.rstrip(f".{file_type}")
        # Check that the wav file has timestamps
        if os.path.isfile(root_name + ".txt"):
            # Parse timestamps
            timestamps = parse_timestamps(root_name + ".txt")
            signal = None
            t = None
            if parsed_args.wav_flag:
                # Open .wav file
                w = wave.open(name)
                # Extract raw audio from .wav file
                signal = (
                    np.array(
                        np.frombuffer(w.readframes(-1), dtype=np.int16),
                        dtype=np.float64,
                    )
                    / 10
                ).tolist()
                # Generate corresponding times
                t = np.linspace(0, len(signal) / w.getframerate(), len(signal))
                # Close the .wav file
                w.close()
            else:
                signal = np.load(name)
                t = np.linspace(0, len(signal) / 10000, len(signal))

            # Create snippets
            event_count = {}
            # Noise snippets indices
            noise_timestamps = []
            for timestamp_id, timestamp in timestamps:
                # Snippet and right proportion
                if config is not None:
                    right_proportion = config[timestamp_id]["right_proportion"]
                    snippet_size = config[timestamp_id]["snippet_size"]
                else:
                    right_proportion = default_right_proportion
                    snippet_size = default_snippet_size

                # Get the index of the minimum time
                min_time_index = np.argmin(
                    np.abs(t - (timestamp - (1 - right_proportion) * snippet_size))
                )
                # Get the index of the maximum time
                max_time_index = np.argmin(
                    np.abs(t - (timestamp + right_proportion * snippet_size))
                )

                # Create slices
                signal_slice = np.array(signal[min_time_index:max_time_index])

                # Apply filter
                filter_flag = True
                if filter_flag:
                    # Apply 50 Hz notch filter to the standardised signal
                    signal_slice = scipy.signal.lfilter(
                        filter_b,
                        filter_a,
                        normalise_signal(signal_slice),
                    )
                # or leave as is
                else:
                    # Standardise signal
                    signal_slice = normalise_signal(signal_slice)

                time_slice = t[min_time_index:max_time_index]
                # Increment the number of events processed for this .wav file
                if timestamp_id not in event_count:
                    event_count[timestamp_id] = 1
                else:
                    event_count[timestamp_id] += 1
                # Save signal and time slices
                np.save(
                    f"{output_directory}/{tail}_{timestamp_id}_{event_count[timestamp_id]}.npy",
                    np.vstack((signal_slice, time_slice)),
                )

                # Plot signal and time if flag
                if parsed_args.plot_flag:
                    # Create new plot
                    plt.figure()
                    # Plot data
                    plt.plot(time_slice, signal_slice)
                    # Axes, labels and title
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.title(
                        f"{tail} Event: {timestamp_id} #{event_count[timestamp_id]}"
                    )
                    # Save figure
                    plt.savefig(
                        f"{output_directory}/Plots/{tail}_{timestamp_id}_{event_count[timestamp_id]}.png"
                    )
                    # Close
                    plt.close()
                if parsed_args.noise_flag:
                    noise_timestamps.append((min_time_index, max_time_index))

            # Create noise snippets
            timestamp_id = "noise"
            min_time_index = 0
            for max_time_index, next_min_index in noise_timestamps:
                if min_time_index > max_time_index:
                    continue
                # Create slices
                signal_slice = np.array(signal[min_time_index:max_time_index])
                if len(signal_slice) < num_samples:
                    continue

                # Apply filter
                filter_flag = True
                if filter_flag:
                    # Apply 50 Hz notch filter to the standardised signal
                    signal_slice = scipy.signal.lfilter(
                        filter_b,
                        filter_a,
                        normalise_signal(signal_slice),
                    )
                # or leave as is
                else:
                    # Standardise signal
                    signal_slice = normalise_signal(signal_slice)

                time_slice = t[min_time_index:max_time_index]
                # Increment indices
                min_time_index = next_min_index
                # Increment the number of events processed for this .wav file
                if timestamp_id not in event_count:
                    event_count[timestamp_id] = 1
                else:
                    event_count[timestamp_id] += 1
                # Save signal and time slices
                np.save(
                    # f"{output_directory}/Noise/{tail}_{timestamp_id}_{event_count[timestamp_id]}.npy",
                    f"{output_directory}/{tail}_{timestamp_id}_{event_count[timestamp_id]}.npy",
                    np.vstack((signal_slice, time_slice)),
                )
                # Plot signal and time if flag
                if parsed_args.plot_flag:
                    # Create new plot
                    plt.figure()
                    # Plot data
                    plt.plot(time_slice, signal_slice)
                    # Axes, labels and title
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.title(
                        f"{tail} Event: {timestamp_id} #{event_count[timestamp_id]}"
                    )
                    # Save figure
                    # plt.savefig(f"{output_directory}/Noise/Plots/{tail}_{timestamp_id}_{event_count[timestamp_id]}.png")
                    plt.savefig(
                        f"{output_directory}/Plots/{tail}_{timestamp_id}_{event_count[timestamp_id]}.png"
                    )
                    # Close
                    plt.close()

            # Update progress
            completed[True].append(tail)
        # Else move on to next .wav file
        else:
            # Update progress
            completed[False].append(tail)
            continue

    # Display results
    if parsed_args.verbose:
        print("Successfully created snippets for:\n")
        print(*completed[True], sep="\n")
        print("\nThese recordings do not have specified timestamp files.\n")
        print(*completed[False], sep="\n")

    print("Finished.")


def _parse_args(args: List) -> argparse.Namespace:
    """Parses command line arguments.

    Parameters
    ----------
    args : List
        unparsed command line arguments

    Returns
    -------
    args : argparse.Namespace
        parsed command line arguments
    """

    parser = argparse.ArgumentParser(
        description="This CLI program is used to create waveform snippets."
    )

    parser.add_argument(
        "input_directory",
        nargs="?",
        type=str,
        help="The input directory containing SpikerRecorder data.",
    )

    parser.add_argument(
        "-o",
        "--o",
        "-output",
        "--output",
        nargs="?",
        default="Snippets",
        type=str,
        help="The output directory to save the snippets to.",
        dest="output_directory",
    )

    parser.add_argument(
        "-s",
        "--s",
        "-snippet_size",
        "--snippet_size",
        nargs="?",
        default=1.0,
        type=float,
        help="The size of the snippets in seconds.",
        dest="snippet_size",
    )

    parser.add_argument(
        "-r",
        "--r",
        "-rightproportion",
        "--rightproportion",
        nargs="?",
        default=0.9,
        type=float,
        help="The proportion of snippet that is to the right of the timestamp.",
        dest="right_proportion",
    )

    parser.add_argument(
        "-c",
        "--c",
        "-config",
        "--config",
        nargs="?",
        default=None,
        help="JSON configuration file that determines the snippet size and right proportion for specific events.",
        dest="config",
    )

    parser.add_argument(
        "-v",
        "--v",
        "-verbose",
        "--verbose",
        action="store_true",
        help="Flag to toggle verbose output.",
        dest="verbose",
    )

    parser.add_argument(
        "-p",
        "--p",
        "-plot",
        "--plot",
        action="store_true",
        help="Flag to create plots.",
        dest="plot_flag",
    )

    parser.add_argument(
        "-n",
        "--n",
        "-noise",
        "--noise",
        action="store_true",
        help="Flag to create noise snippets.",
        dest="noise_flag",
    )

    parser.add_argument(
        "-w",
        "--w",
        "-wav",
        "--wav",
        action="store_true",
        help="Set true if the files are wavs.",
        dest="wav_flag",
    )

    return parser.parse_args(args)


def parse_timestamps(filename: str) -> List[Tuple[str, float]]:
    """Parses SpikerRecorder timestamp files.

    Parameters
    ----------
    filename : str
        the name of the timestamp file.

    Returns
    -------
    timestamps : List[Tuple[str, float]]
        a list of timestamps (an ID and a time).
    """
    timestamps = []
    with open(filename) as open_file:
        lines = open_file.readlines()
        for line in lines:
            if "#" not in line:
                timestamp_id, timestamp = line.split(",\t")
                timestamps.append((timestamp_id, float(timestamp)))
    return timestamps


if __name__ == "__main__":
    main()
