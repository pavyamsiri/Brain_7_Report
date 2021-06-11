# Future imports
from __future__ import annotations
from enum import Enum

# Standard modules
import time
from typing import Dict, List, NamedTuple, Optional

# External modules
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from scipy import signal
from serial.serialutil import SerialException
from aliases import FilePath, PortName
from models import Event, ModelBase

# Internal modules
from data_stream import (
    ArrayStream,
    InputStream,
    SpikerStream,
    StreamException,
    StreamType,
    WAVStream,
)
from utils import normalise_signal


# Constants
# Electronic noise frequency to filter out (Hz)
NOISE_FREQUENCY = 50


class Control(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class FilterArgs(NamedTuple):
    """Notch filter parameters."""

    b: np.ndarray
    a: np.ndarray


class SpikerBox:
    """A class that acts as an interface between the Game and the InputStream. It also deals with event classification and other
    diagnostics.

    Attributes
    ----------
    model : ModelBase
        the classifier model.
    stream : InputStream
        the InputStream to read from.
    buffer_time : float
        the size of the moving buffer in seconds.
    update_factor : float
        the fraction of the moving buffer read in each stream update.
    buffer_size : int
        the size of the moving buffer.
    buffer : np.ndarray
        the moving buffer.
    buffer_pointer : int
        the current index of the moving buffer.
    num_samples : int
        the number of samples to downsample the moving buffer to before classification.
    global_time : float
        global clock used in timestamping events.
    read_timer : int
        the stream update timer.
    wait_time : float
        the number of seconds the SpikerBox stops analysing after a classification.
    wait_timer : int
        the wait timer used after classification.
    wait_flag : bool
        if `True` will wait `wait_time` seconds before analysing again.
    analyse_flag : bool
        if `True` will not analyse the signal.
    filter_args : FilterArgs
        the notch filter parameters.
    filter_flag : bool
        if `True` the data will be notch filtered otherwise it will be left unfiltered.
    figure : Optional[pyqtgraph.graphicsWindows.PlotWindow]
        the PlotWindow.
    plot_flag : bool
        if `True` the data plotted every frame.
    graph_path : Optional[FilePath]
        the file path to the graph snapshot.
    restart_flag : bool
        if `True` the InputStream is restarting to the beginning of the file if applicable.
    """

    def __init__(
        self,
        model: ModelBase,
        buffer_time: float,
        update_factor: float,
        wait_time: float,
        num_samples: int,
        quality_factor: float,
        stream_type: StreamType,
        stream_file: Optional[FilePath],
        cport: Optional[PortName],
        filter_flag: bool = True,
        plot_flag: bool = True,
    ):
        """Initialises the SpikerBox input stream.

        Parameters
        ----------
        model : ModelBase
            the classifier model.
        buffer_time : float
            the size of the moving buffer in seconds.
        update_factor : float
            the fraction of the moving buffer read in each stream update.
        wait_time : float
            the number of seconds the SpikerBox stops analysing after a classification.
        num_samples : int
            the number of samples to downsample the moving buffer to before classification.
        quality_factor : float
            the quality factor of the Notch filter.
        stream_type : StreamType
            the type of input stream used.
        stream_file : Optional[FilePath]
            the file path to the stream file that some InputStreams might require.
        cport : Optional[PortName]
            the serial port name to the SpikerBox. This required if using SpikerStream.
        filter_flag : bool
            if `True` the data will be notch filtered otherwise it will be left unfiltered.
        plot_flag : bool
            if `True` the data plotted every frame.
        """
        # Classifier model
        self._model: ModelBase = model

        # Size of moving buffer in seconds
        self._buffer_time: float = buffer_time
        # Fraction of moving buffer size to read from InputStream
        self._update_factor: float = update_factor

        # Initialise streams
        try:
            self._stream: InputStream = self._initialise_stream(
                stream_type, stream_file, cport
            )
        except (SerialException, StreamException) as e:
            raise e

        # Initialise buffer
        self._buffer_size: int = int(self._stream.chunk / self._update_factor)
        self._buffer: np.ndarray = np.zeros(self._buffer_size)
        self._buffer_pointer: int = 0
        # Number of samples to downsample to
        self._num_samples: int = num_samples

        # Global clock for timestamping events
        self._global_time: float = time.time()
        self._timestamps: List[float] = []
        # Stream update timer
        self._read_timer: int = 0
        # Wait timer
        self._wait_time: float = wait_time
        self._wait_timer: int = 0
        self._wait_flag: bool = False

        # Analyse flag
        self._analyse_flag: bool = False

        # Filter
        filter_b, filter_a = signal.iirnotch(
            NOISE_FREQUENCY, quality_factor, fs=self._stream.sample_rate
        )
        self._filter_args: FilterArgs = FilterArgs(filter_b, filter_a)
        # Filter flag
        self._filter_flag: bool = filter_flag

        # Plot
        self._figure: Optional[pyqtgraph.graphicsWindows.PlotWindow] = None
        # Plot flag
        self._plot_flag: bool = plot_flag
        # FilePath to graph snapshot
        self._graph_path: Optional[FilePath] = None

        # Stream restart flag
        self._restart_flag: bool = False

    def _initialise_stream(
        self,
        stream_type: StreamType,
        stream_file: Optional[FilePath],
        cport: Optional[PortName],
    ) -> InputStream:
        """Initialises the input stream.

        Parameters
        ----------
        stream_type : StreamType
            the type of input stream to initialise.
        stream_file : Optional[FilePath]
            the path to the stream file if using a recorded stream.
        cport : Optional[PortName]
            the name of the serial port of the SpikerBox if using SpikerStream.

        Returns
        -------
        input_stream : InputStream
            the initialised input stream.

        Raises
        ------
        StreamException
            Raised if
            - the serial port is not given if using SpikerStream.
            - the stream file is not given or invalid if using either ArrayStream or WAVStream.
            - no stream type was given or the stream type was invalid.
        SerialException
            Raised if the given serial port is invalid.
        """
        if stream_type == StreamType.SPB:
            if cport is None:
                raise StreamException("No serial port given!")
            else:
                try:
                    return SpikerStream(
                        cport,
                        self._update_factor * self._buffer_time,
                    )
                except SerialException as e:
                    raise e
        elif stream_file is None:
            raise StreamException("No stream file given!")
        elif stream_type == StreamType.ARR:
            try:
                return ArrayStream(
                    stream_file,
                    self._update_factor * self._buffer_time,
                )
            except StreamException as e:
                raise e
        elif stream_type == StreamType.WAV:
            try:
                return WAVStream(stream_file, self._update_factor * self._buffer_time)
            except StreamException as e:
                raise e
        elif stream_type == StreamType.NIL:
            raise StreamException("No stream given!")
        else:
            raise StreamException(f"Invalid stream type {stream_type}!")

    def reset_clock(self):
        """Resets the global clock."""
        self._global_time = time.time()

    @property
    def graph_path(self) -> Optional[FilePath]:
        """Optional[FilePath]: the file path to the graph snapshot."""
        return self._graph_path

    @property
    def timestamps(self) -> List[float]:
        """List[float]: the event timestamps."""
        return self._timestamps

    def can_move(self) -> bool:
        """Returns whether or not the player can move.

        Returns
        -------
        `True` if the signal is being analysed and hence the player can move else `False`.
        """
        return self._analyse_flag

    def is_restarting(self) -> bool:
        """Returns whether or not the InputStream is restarting.

        Returns
        -------
        `True` if the InputStream is restarting else `False`.
        """
        return self._restart_flag

    def process_input(
        self, controls: Dict[Control, int], tick: int
    ) -> Dict[Control, int]:
        """Processes data from SpikerBox to change control parameters using the given classifier model.

        Parameters
        ----------
        controls : Dict[Control, int]
            the control parameters.
        tick : int
            the number of milliseconds since last frame.

        Returns
        -------
        controls : Dict[Control, int]
            the changed control parameters.
        """

        # If the stream has restarted to the beginning of the stream file
        if self._restart_flag:
            # Reset global timer
            self._global_timer = time.time()

        # Update timers
        self._read_timer += tick
        if self._wait_flag:
            self._wait_timer += tick
        # Wait at least `buffer_time` seconds before reading from stream
        if self._read_timer / 1000 < self._stream.buffer_time:
            return controls
        else:
            self._read_timer = 0

        # Read from stream
        raw_data = self._stream.update()
        # Update restart flag
        self._restart_flag = self._stream.is_restarting()

        # Wait at least `wait_time` seconds before analysing the buffer again
        if self._wait_flag and self._wait_timer / 1000 >= self._wait_time:
            self._wait_timer = 0
            self._wait_flag = False
        elif not self._wait_flag:
            self._wait_timer = 0

        # Moving window buffer
        shift: int = self._buffer_pointer + len(raw_data) - self._buffer_size
        # Shift the buffer to make room for new data
        if shift > 0:
            self._buffer = np.roll(self._buffer, -shift)
            self._buffer[
                len(self._buffer) - len(raw_data) : len(self._buffer)
            ] = raw_data
            self._analyse_flag = not self._wait_flag
        # Fill the emptied buffer with data
        else:
            self._buffer[
                self._buffer_pointer : self._buffer_pointer + len(raw_data)
            ] = raw_data
            self._buffer_pointer += len(raw_data)
            self._analyse_flag = False

        signal_slice: np.ndarray
        # Filter signal
        if self._filter_flag:
            # Apply 50 Hz notch filter to the standardised signal
            signal_slice = signal.lfilter(
                self._filter_args.b, self._filter_args.a, normalise_signal(self._buffer)
            )
        # or leave as is
        else:
            # Standardise signal
            signal_slice = normalise_signal(self._buffer)

        # Downsample signal
        processed_data: np.ndarray = signal.resample(signal_slice, self._num_samples)

        # Current time just before classification
        current_time: float = time.time() - self._global_time
        label: Optional[Event] = None
        # Only classify filled buffers
        if self._analyse_flag:
            label = self._model.classify(processed_data)

        # Initialise plot
        if self._figure is None and self._plot_flag:
            self._figure = pg.plot(
                np.linspace(
                    current_time - self._buffer_time, current_time, len(processed_data)
                ),
                processed_data,
                labels={
                    "left": "Standardised amplitude (a.u.)",
                    "bottom": "Time (s)",
                },
                title="Moving buffer",
            )
            max_amplitude = np.max(np.abs(processed_data))
            self._figure.setXRange(current_time - self._buffer_time, current_time)
            self._figure.setYRange(-1.2 * max_amplitude, 1.2 * max_amplitude)
        # Update plot
        elif self._figure is not None and self._plot_flag:
            self._figure.plot(
                np.linspace(
                    current_time - self._buffer_time, current_time, len(processed_data)
                ),
                processed_data,
                clear=True,
            )
            max_amplitude = np.max(np.abs(processed_data))
            self._figure.setXRange(current_time - self._buffer_time, current_time)
            self._figure.setYRange(-1.2 * max_amplitude, 1.2 * max_amplitude)
            pg.QtGui.QApplication.processEvents()

        # Event detected
        if label is not None and label != Event.NOISE:
            # Diagnostics
            self._timestamps.append(current_time)
            # Export snapshot
            if self._figure is not None:
                exporter = pg.exporters.ImageExporter(self._figure.plotItem)
                self._graph_path = f"snapshots/{str(label)}.png"
                exporter.export(self._graph_path)
            # Export data
            np.save(f"snapshots/{str(label)}.npy", raw_data)

            # Clear buffer and wait `wait_time` seconds before analysing again
            self._buffer[:] = 0
            self._wait_flag = True
            self._buffer_pointer = 0
            # Set control parameters
            controls[Control.LEFT] = 1 if label == Event.LEFT else 0
            controls[Control.RIGHT] = 1 if label == Event.RIGHT else 0

        # Return control parameters
        return controls
