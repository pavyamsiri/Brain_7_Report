# Standard modules
import argparse
import glob
import sys
import time
from typing import Any, Dict, List, Tuple

# External modules
from catch22 import catch22_all
from joblib import dump
import numpy as np
from scipy import signal
import yaml

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Internal modules
from aliases import FilePath
from models import Event
from utils import get_snippet_event, parse_snippet


# Type aliases
Snippets = Dict[str, List[Tuple[np.ndarray, np.ndarray]]]


# Constants
CONFIG_PATH: FilePath = "settings/spiker_box.srconfig"


def main(args: List = sys.argv[1:]):
    """Main entry point.

    Parameters
    ----------
    args : List
        unparsed command line arguments
    """
    parsed_args = _parse_args(args)

    # Select model to train
    models: List[Any] = []
    if parsed_args.model_type == "KNN":
        models.append(KNeighborsClassifier(n_neighbors=5))
    elif parsed_args.model_type == "RFC":
        models.append(RandomForestClassifier())
    elif parsed_args.model_type == "SVC":
        models.append(svm.SVC())
    elif parsed_args.model_type == "all":
        models.append(KNeighborsClassifier(n_neighbors=5))
        models.append(RandomForestClassifier())
        models.append(svm.SVC())
    else:
        print(
            f"The model type {parsed_args.model_type} is invalid! Please choose a valid model type."
        )
        sys.exit()

    # Load configuration files
    with open(CONFIG_PATH, "r") as config_file:
        config_data: Dict = yaml.safe_load(config_file)
        num_samples: int = int(config_data["num_samples"])

    # Load snippets
    print("Loading snippets...")
    snippets = load_snippets({}, parsed_args.snippet_folder)
    print("Loading complete!")
    print("Loading noise snippets...")
    snippets = load_snippets(snippets, parsed_args.noise_folder)
    print("Loading complete!")

    start_time: float = time.time()
    # Process snippets into catch22 data and labels
    print("Processing snippets...")
    data, labels = process_snippets(snippets, num_samples)
    print("Processing complete!")

    # Train model
    print("Training...")
    for model in models:
        model.fit(data, labels)
    print("Training complete!")
    elapsed_time = time.time() - start_time
    print(f"This took {elapsed_time:.2f} seconds to complete!")

    # Save model
    if len(models) == 1:
        dump(models[0], f"models/{parsed_args.model_type}.joblib")
    else:
        dump(models[0], f"models/KNN.joblib")
        dump(models[1], f"models/RFC.joblib")
        dump(models[2], f"models/SVC.joblib")


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
        description="This CLI program is used to train models."
    )

    parser.add_argument(
        "model_type",
        choices=["KNN", "RFC", "SVC", "all"],
        help="The type of model to train.",
    )

    parser.add_argument(
        "-s",
        "--s",
        "-snippets",
        "--snippets",
        type=FilePath,
        default="snippets",
        help="The folder containing the event snippets.",
        dest="snippet_folder",
    )

    parser.add_argument(
        "-n",
        "--n",
        "-noise",
        "--noise",
        type=FilePath,
        default="snippets",
        help="The folder containing the noise snippets.",
        dest="noise_folder",
    )

    return parser.parse_args(args)


def load_snippets(snippets: Snippets, snippet_folder: FilePath) -> Snippets:
    """Load snippets from a folder into a dictionary of event types and slices.

    Parameters
    ----------
    snippets : Snippets
        a dictionary of events and a list of slices.
    snippet_folder : FilePath
        the folder of snippets to load from.

    Returns
    -------
    snippets : Snippets
        the loaded snippets.
    """
    # Find all snippets
    for snippet_file in glob.glob(f"{snippet_folder}/*.npy"):
        # Load data
        snippet: np.ndarray = np.load(snippet_file)
        signal_slice, time_slice = parse_snippet(snippet)
        if len(signal_slice) == 0:
            continue
        event = get_snippet_event(snippet_file)
        # Looking only at left, right and noise
        if event != "left" and event != "right" and event != "noise":
            continue
        # Add data to dictionary
        if event not in snippets:
            snippets[event] = [(signal_slice, time_slice)]
        else:
            snippets[event].append((signal_slice, time_slice))

        # Cheeky trick
        if event == "left":
            if "right" not in snippets:
                snippets["right"] = [(-signal_slice, time_slice)]
            else:
                snippets["right"].append((-signal_slice, time_slice))

        if event == "right":
            if "left" not in snippets:
                snippets["left"] = [(-signal_slice, time_slice)]
            else:
                snippets["left"].append((-signal_slice, time_slice))

    return snippets


def load_snippet_files(snippets: Snippets, file_paths: List[FilePath]) -> Snippets:
    """Load snippets from a folder into a dictionary of event types and slices.

    Parameters
    ----------
    snippets : Snippets
        a dictionary of events and a list of slices.
    file_paths : List[FilePath]
        a list of file paths to the snippets to train on.

    Returns
    -------
    snippets : Snippets
        the loaded snippets.
    """
    # Find all snippets
    for snippet_file in file_paths:
        # Load data
        snippet: np.ndarray = np.load(snippet_file)
        signal_slice, time_slice = parse_snippet(snippet)
        if len(signal_slice) == 0:
            continue
        event = get_snippet_event(snippet_file)
        # Looking only at left, right and noise
        if event != "left" and event != "right" and event != "noise":
            continue
        # Add data to dictionary
        if event not in snippets:
            snippets[event] = [(signal_slice, time_slice)]
        else:
            snippets[event].append((signal_slice, time_slice))

        # Cheeky trick
        if event == "left":
            if "right" not in snippets:
                snippets["right"] = [(-signal_slice, time_slice)]
            else:
                snippets["right"].append((-signal_slice, time_slice))

        if event == "right":
            if "left" not in snippets:
                snippets["left"] = [(-signal_slice, time_slice)]
            else:
                snippets["left"].append((-signal_slice, time_slice))

    return snippets


def process_snippets(snippets: Snippets, num_samples: int) -> Tuple[List, List[str]]:
    """Process snippet data into a list of catch22 features.

    Parameters
    ----------
    snippets : Snippets
        the snippets to process.
    num_samples : int
        the number of samples to downsample to.

    Returns
    -------
    data : List[List]
        list of catch22 features for each snippet.
    labels : List[str]
        corresponding list of labels for each snippet.
    """
    # Compute catch22 data
    data: List[Any] = []
    # Create list of labels and names associated with catch22 data
    labels: List[str] = []
    for event in snippets:
        for signal_slice, _ in snippets[event]:
            signal_slice: np.ndarray = signal.resample(signal_slice, num_samples)
            data.append(catch22_all(signal_slice)["values"])
            labels.append(event)
    print(f"There are {len(data)} samples to train/test on.")

    return data, labels


if __name__ == "__main__":
    main()
