# Standard modules
import os
from typing import Tuple

# External modules
import numpy as np

# Internal modules


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    """Normalises a signal.

    Parameters
    ----------
    signal : np.ndarray
        the given signal to normalise.

    Returns
    -------
    normalised_signal : np.ndarray
        the normalised signal.
    """
    signal_centre = np.mean(signal)
    standardised_signal = signal - signal_centre
    # signal_amplitude = np.max(np.abs(standardised_signal))
    return standardised_signal


def parse_snippet(snippet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parses snippet files into signal and time slices.

    Parameters
    ----------
    snippet : np.ndarray
        the array loaded from a snippet file.

    Returns
    -------
    signal_slice : np.ndarray
        the normalised amplitude of the snippet.
    time_slice : np.ndarray
        the time slice of the snippet.
    """
    signal_slice: np.ndarray
    time_slice: np.ndarray
    if np.shape(snippet)[0] == 2:
        signal_slice = snippet[0, :]
        time_slice = snippet[1, :]
    else:
        signal_slice = snippet
        time_slice = np.linspace(
            0,
            len(signal_slice) / 10000,
            len(signal_slice),
        )
    return signal_slice, time_slice


def get_snippet_event(snippet_filename: str) -> str:
    """Get the event name of the snippet located at `snippet_filename`.

    Parameters
    ----------
    snippet_filename : str
        the path to a snippet file created by WaveformSnipper. Must be in the correct name format.

    Returns
    -------
    event_name : str
        the event name of the snippet.
    """
    _, tail = os.path.split(snippet_filename)
    return tail.split("_")[-2]
