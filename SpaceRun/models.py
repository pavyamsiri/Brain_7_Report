"""
Classifier model interface with two distinct implementations.
1. Simple classifier - requires no training data and uses heuristics to classify.
2. catch22 models - uses catch22 to compute features of the waveform to use in machine learning algorthims.
"""
# Future imports
from __future__ import annotations

# Standard modules
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, List, Protocol

# External modules
from catch22 import catch22_all
import joblib
import numpy as np

# Internal modules
from aliases import FilePath


class Event(Enum):
    """Classification labels"""

    LEFT = "left"
    RIGHT = "right"
    NOISE = "noise"

    def __str__(self) -> str:
        return self.value


class ModelException(Exception):
    """Exceptions to do with ClassifierModels"""

    ...


class ModelType(Enum):
    """Type of classifier model."""

    # k-nearest neighbours
    KNN = "KNN"
    # Random forest classifier with 100 trees
    RFC = "RFC"
    # C-Support vector classification
    SVC = "SVC"
    # Modified simple classifier
    MSC = "MSC"
    # Unmodified simple classifier
    USC = "USC"
    # No model
    NIL = "NIL"
    # Default model as defined in spiker_box.srconfig
    DEFAULT = "DEFAULT"

    def __str__(self):
        return self.value


class ModelBase(metaclass=ABCMeta):
    """Classifier model interface. Needs to implement the `classify` method."""

    def __init__(self):
        """Initialises the classifier model.

        Implement for each class inheriting from this base class.
        """
        pass

    @abstractmethod
    def classify(self, _signal_slice: np.ndarray) -> Event:
        """Classifies a slice of data giving it a label.

        Parameters
        ----------
        signal_slice : np.ndarray
            signal slice to classify.

        Returns
        -------
        label : Event
            the detected event or noise.

        Implement for each class inheriting from this base class.
        """
        raise NotImplementedError("`classify` is not implemented!")


class SupportsPredict(Protocol):
    def predict(self, x: List[np.ndarray]) -> np.ndarray:
        ...


class Catch22Model(ModelBase):
    """Interface class for models that use catch22 data and pickled sklearn classifiers.

    Attributes
    ----------
    model : SupportsPredict
        the machine learning model trained on catch22 data.
    """

    def __init__(self, model: SupportsPredict):
        """Initialises the Catch22 model by loading a trained cached model.

        Parameters
        ----------
        model : SupportsPredict
            the trained machine learning model.
        """
        self._model: SupportsPredict = model

    def classify(self, signal_slice: np.ndarray) -> Event:
        # Compute features
        data: np.ndarray = catch22_all(signal_slice)["values"]
        # Predict movement
        label_str: str = self._model.predict([data])[0]
        label: Event = Event(label_str)
        return label

    @staticmethod
    def create_from_file(model_path: FilePath) -> Catch22Model:
        """Initialises the Catch22 model by loading a trained cached model.

        Parameters
        ----------
        model_path : FilePath
            the file path to the trained cached model.

        Returns
        -------
        model : Catch22Model
            the trained cached model.
        """
        return Catch22Model(joblib.load(model_path))


class SimpleModel(ModelBase):
    """Simple classifier model. Uses zero-crossing method to detect
    and then classifies using ordering of peak and trough.

    Attributes
    ----------
    event_threshold : int
        the number of zero-crossing events until the signal is interpreted as noise.
    """

    def __init__(self, event_threshold: int):
        """Initialises the simple classifier model.

        Parameters
        ----------
        event_threshold : int
            the number of zero-crossing events until the signal is interpreted as noise.
        """
        self._event_threshold = event_threshold

    def classify(self, signal_slice: np.ndarray) -> Event:
        """Simple classifier model. Uses zero-crossing method to detect
        and then classifies using ordering of peak and trough.

        Parameters
        ----------
        signal_slice : np.ndarray
            the signal to classify.

        Returns
        -------
        event : Event
            the classification.
        """
        # Signal subset
        sub_signal: np.ndarray = signal_slice[0 : len(signal_slice) - 1]
        # Offset signal subset
        off_signal: np.ndarray = signal_slice[1 : len(signal_slice) - 0]

        # Many different signs pairs imply the signal varies around 0 often
        # Test stat is the number of crossing events
        test_stat: int = 0
        for num1, num2 in zip(sub_signal, off_signal):
            test_stat += 1 if num1 * num2 <= 0 else 0

        # Events are when the signal crosses the zero line less than `event_threshold`.
        if test_stat < self._event_threshold:
            return SimpleModel._left_right_detection(signal_slice)
        # Noise event
        else:
            return Event.NOISE

    @staticmethod
    def _left_right_detection(signal_slice: np.ndarray) -> Event:
        """Classifying a signal between left and right eye movements.

        Parameters
        ----------
        signal_slice : np.ndarray
            the signal to analyse.

        Returns
        -------
        event : Event
            the event classification, either left or right.
        """
        # Indices of the maximum and minimum
        max_index: np.integer[Any] = np.argmax(signal_slice)
        min_index: np.integer[Any] = np.argmin(signal_slice)

        # Left-right classification
        if max_index > min_index:
            return Event.LEFT
        else:
            return Event.RIGHT


class ModifiedModel(ModelBase):
    """Modified simple classifier model. Uses zero-crossing method to detect
    and then classifies using ordering of peak and trough. It includes extra restrictions to be able to better distinguish
    noise from an eye movement.

    Attributes
    ----------
    event_threshold : int
        the number of zero-crossing events until the signal is interpreted as noise.
    positive_amplitude : float
        the minimum amplitude threshold required for a peak to be counted as a non-noise event. Has a positive value.
    negative_amplitude : float
        the minimum amplitude threshold required for a trough to be counted as a non-noise event. Has a postive value.
    spacing : float
        the maximum spacing allowed for the peak and trough. Stored as a fraction of the signal_slice.
    """

    def __init__(
        self,
        event_threshold: int,
        positive_amplitude: float,
        negative_amplitude: float,
        spacing: float,
    ):
        """Initialises the modified simple classifier model.

        Parameters
        ----------
        event_threshold : int
            the number of zero-crossing events until the signal is interpreted as noise.
        positive_amplitude : float
            the minimum amplitude threshold required for a peak to be counted as a non-noise event. Has a positive value.
        negative_amplitude : float
            the minimum amplitude threshold required for a trough to be counted as a non-noise event. Has a postive value.
        spacing : float
            the maximum spacing allowed for the peak and trough. Stored as a fraction of the signal_slice.
        """
        self._event_threshold: int = event_threshold
        self._positive_amplitude: float = positive_amplitude
        self._negative_amplitude: float = negative_amplitude
        self._spacing: float = spacing

    def classify(self, signal_slice: np.ndarray) -> Event:
        """Modified simple classifier model. Uses zero-crossing method to detect
        and then classifies using ordering of peak and trough. It includes extra restrictions to be able to better distinguish
        noise from an eye movement.

        Parameters
        ----------
        signal_slice : np.ndarray
            the signal to classify.

        Returns
        -------
        event : Event
            the classification.
        """
        # Signal subset
        sub_signal: np.ndarray = signal_slice[0 : len(signal_slice) - 1]
        # Offset signal subset
        off_signal: np.ndarray = signal_slice[1 : len(signal_slice) - 0]

        # Many different signs pairs imply the signal varies around 0 often
        # Test stat is the number of crossing events
        test_stat: int = 0
        for num1, num2 in zip(sub_signal, off_signal):
            test_stat += 1 if num1 * num2 <= 0 else 0

        # Events are when the signal crosses the zero line less than `event_threshold`.
        if test_stat < self._event_threshold:
            return self._left_right_detection(signal_slice)
        else:
            return Event.NOISE

    def _left_right_detection(self, signal_slice: np.ndarray) -> Event:
        """Classifying a signal between left and right eye movements.
        There are extra restrictions to avoid misclassifying noise events.

        Parameters
        ----------
        signal_slice : np.ndarray
            the signal to analyse.

        Returns
        -------
        event : Event
            the event classification, either left, right or noise.
        """
        # Check if amplitudes reach the minimum thresholds
        if (
            np.max(signal_slice) < self._positive_amplitude
            or np.min(signal_slice) > -self._negative_amplitude
        ):
            return Event.NOISE

        # Indices of the maximum and minimum
        max_index: np.integer[Any] = np.argmax(signal_slice)
        min_index: np.integer[Any] = np.argmin(signal_slice)

        # Check if the peak and trough are close enough
        if np.abs(max_index - min_index) > int(self._spacing * len(signal_slice)):
            return Event.NOISE

        # Left-right classification
        if max_index > min_index:
            return Event.LEFT
        else:
            return Event.RIGHT
