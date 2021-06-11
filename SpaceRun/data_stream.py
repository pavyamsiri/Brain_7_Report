# Future imports
from __future__ import annotations

# Standard modules
from abc import ABCMeta, abstractmethod
from enum import Enum
import wave

# External modules
import numpy as np
import serial
from serial.serialutil import SerialException
from serial.tools import list_ports

# Internal modules
from aliases import PortName, FilePath


# Constants
# 20,000 samples correspond to a chunk of 1 second.
CHUNK_UNIT_TIME = 20000
# Processing reduces this to 10,000 samples as there is padding.
ARRAY_UNIT_SIZE = CHUNK_UNIT_TIME // 2


class StreamType(Enum):
    # SpikerBox stream
    SPB = "SPB"
    # numpy array stream
    ARR = "ARR"
    # WAV stream
    WAV = "WAV"
    # No stream
    NIL = "NIL"
    # Default stream as defined in spiker_box.srconfig
    DEFAULT = "DEFAULT"

    def __str__(self):
        return self.value


class StreamException(Exception):
    ...


class InputStream(metaclass=ABCMeta):
    """Input stream base class.

    Attributes
    ----------
    chunk : int
        the number of samples returned by the `update` method.
    sample_rate : float
        the average number of samples read per second.
    buffer_time : float
        the amount of time captured by the `update` method.
    """

    def __init__(self, chunk: int, sample_rate: float):
        """Initialises the input stream.

        Implement for each class inheriting from this base class.
        """
        self._chunk: int = chunk
        self._sample_rate: float = sample_rate
        self._buffer_time: float = chunk / self.sample_rate

    @property
    def chunk(self) -> int:
        """int: the number of samples returned by the `update` method."""
        return self._chunk

    @property
    def sample_rate(self) -> float:
        """float: the average number of samples read per second."""
        return self._sample_rate

    @property
    def buffer_time(self) -> float:
        """float: the amount of time captured by the `update` method."""
        return self._buffer_time

    @abstractmethod
    def update(self) -> np.ndarray:
        """Reads data from the input stream.

        Returns
        -------
        data : np.ndarray
            data read from the input stream.

        Implement for each class inheriting from this base class.
        """
        raise NotImplementedError("`update` method is not implemented!")

    @abstractmethod
    def close(self):
        """Closes the input stream.

        Implement for each class inheriting from this base class.
        """
        raise NotImplementedError("`close` method is not implemented!")

    def is_restarting(self) -> bool:
        """Returns whether or not the stream is restarting.

        Returns
        -------
        restart_flag : bool
            `True` if stream is restarting else `False`. If it doesn't make sense for the stream to restart then it is `False`.
        """
        return False


class SpikerStream(InputStream):
    """Streams in data from Backyard Brains' SpikerBox.

    Attributes
    ----------
    serial_handle : serial.Serial
        a handle to the serial port the SpikerBox is connected to.
    chunk : int
        the number of samples returned by the `update` method.
    sample_rate : float
        the average number of samples read per second.
    buffer_time : float
        the amount of time captured by the `update` method.
    """

    # Baud rate is the rate of change of symbols (i.e. for cases where the data is non-binary)
    BAUDRATE = 230400

    def __init__(self, cport: PortName, buffer_time: float):
        """Initialises and opens the input stream.

        Parameters
        ----------
        cport : PortName
            the name of the serial port the SpikerBox is connected to.
        buffer_time : float
            the size of the input buffer in seconds. Note 20 000 = 1 second of time.

        Raises
        ------
        StreamException
            Raised if serial port can not be opened.
        """
        # Create a serial handle
        try:
            print(f"Attempting to open port {cport}\n")
            self._serial_handle: serial.Serial = serial.Serial(
                port=cport, baudrate=SpikerStream.BAUDRATE
            )
        # Otherwise list serial ports and exit
        except SerialException as _:
            print("The list of serial ports:")
            ports = list_ports.comports()
            for idx, port in enumerate(ports):
                print(f"\t{idx+1}: {port}")
            raise SerialException("Can not open serial port!")

        # Chunk size and sample rate
        super().__init__(int(ARRAY_UNIT_SIZE * buffer_time), ARRAY_UNIT_SIZE)

        # Set read timeout
        self._serial_handle.timeout = buffer_time

    def update(self) -> np.ndarray:
        """Reads the input stream from the arduino/SpikerBox.

        Returns
        -------
        data : np.ndarray
            streamed processed data coming from the SpikerBox of size `chunk`.
        """
        data: np.ndarray = self._read_arduino()
        processed_data: np.ndarray = SpikerStream._process_data(data)
        return processed_data

    def close(self) -> None:
        """Closes the input stream by closing the serial port."""

        # Close the serial port
        if self._serial_handle.read():
            self._serial_handle.flushInput()
            self._serial_handle.flushOutput()
            self._serial_handle.close()

    def _read_arduino(self) -> np.ndarray:
        """Read the input buffer.

        Returns
        -------
        data : np.ndarray
            the raw data from the input buffer read as integers.
        """
        raw_data: bytes = self._serial_handle.read(self._chunk)
        int_data = [int(data_bit) for data_bit in raw_data]
        return np.array(int_data)

    @staticmethod
    def _process_data(data: np.ndarray) -> np.ndarray:
        """Processes the raw data from the input buffer.

        Parameters
        ----------
        data : np.ndarray
            unprocessed input buffer data.

        Returns
        -------
        result : np.ndarray
            processed input buffer data.
        """
        result: np.ndarray = np.empty(shape=(0, 0))
        i = 0
        while i < (len(data) - 1):
            # Found beginning of frame
            if data[i] > 127:
                # Extract one sample from 2 bytes
                intout = (np.bitwise_and(data[i], 127)) * 128
                i += 1
                intout = intout + data[i]
                result = np.append(result, intout)
            i += 1
        return result


class ArrayStream(InputStream):
    """Streams in data loaded from a numpy array file.

    Attributes
    ----------
    signal_data : np.ndarray
        the recorded SpikerStream data in numpy array form.
    time_data : np.ndarray
        the times corresponding to each sample.
    chunk : int
        the number of samples returned by the `update` method.
    sample_rate : float
        the average number of samples read per second.
    buffer_time : float
        the amount of time captured by the `update` method.
    pointer : int
        the current index of the stream.
    restart_flag : bool
        `True` if stream is restarting else `False`.
    """

    def __init__(self, array_path: FilePath, buffer_time: float):
        """Initialises the input stream.

        Parameters
        ----------
        array_path: FilePath
            the path of the array to stream in.
        buffer_time : float
            the size of the input buffer in seconds. Note 20 000 = 1 second of time.
        """
        # Load array from `array_path`
        try:
            loaded_data: np.ndarray = np.load(array_path)
        except FileNotFoundError as _:
            raise StreamException(f"Stream file {array_path} not found!")

        self._signal_data: np.ndarray
        self._time_data: np.ndarray
        # Recording stores time as well as data
        if np.shape(loaded_data)[0] == 2:
            self._signal_data = loaded_data[0, :]
            self._time_data = loaded_data[1, :]
        # Recording only stores data
        else:
            self._signal_data = loaded_data
            self._time_data = np.linspace(
                0,
                len(self._signal_data) / ARRAY_UNIT_SIZE,
                len(self._signal_data),
            )

        # Chunk size and sample rate
        super().__init__(int(ARRAY_UNIT_SIZE * buffer_time), ARRAY_UNIT_SIZE)
        # Index of stream
        self._pointer: int = 0

        # Restart flag
        self._restart_flag: bool = False

    def update(self) -> np.ndarray:
        """Reads data from the input buffer.

        Returns
        -------
        signal_slice : np.ndarray
            next chunk of stream.
        """
        # Read chunk from array
        signal_slice: np.ndarray = np.array(
            self._signal_data[self._pointer : self._pointer + self._chunk]
        )
        # Move index over
        self._pointer += self._chunk
        if self._pointer > len(self._signal_data):
            # Go back to beginning
            self._pointer = 0
            self._restart_flag = True
            print("Restarting stream...")

        return signal_slice

    def close(self):
        """Closes the input stream."""
        pass

    def is_restarting(self) -> bool:
        """Returns whether the stream is restarting or not.

        Returns
        -------
        restart_flag : bool
            `True` if stream is restarting else `False`.
        """
        restart_flag = self._restart_flag
        # Reset the flag
        self._restart_flag = False

        return restart_flag


class WAVStream(InputStream):
    """Streams in data loaded from a numpy array file.

    Attributes
    ----------
    signal_data : np.ndarray
        the recorded SpikerRecorder data in numpy array form.
    time_data : np.ndarray
        the times corresponding to each sample.
    chunk : int
        the number of samples returned by the `update` method.
    sample_rate : float
        the average number of samples read per second.
    buffer_time : float
        the amount of time captured by the `update` method.
    pointer : int
        the current index of the stream.
    restart_flag : bool
        `True` if stream is restarting else `False`.
    """

    def __init__(self, wav_path: FilePath, buffer_time: float):
        """Initialises the input stream.

        Parameters
        ----------
        wav_path: FilePath
            the path of the wav file to stream in.
        buffer_time : float
            the size of the input buffer in seconds. Note 20 000 = 1 second of time.
        """
        # Open .wav file
        try:
            w = wave.open(wav_path)
        except FileNotFoundError as _:
            raise StreamException(f"Stream file {wav_path} not found!")
        # Extract raw audio from .wav file TODO: seems to be a list instead
        self._signal_data: np.ndarray = np.array(
            np.frombuffer(w.readframes(-1), dtype=np.int16), dtype=np.float64
        ).tolist()
        # Frame rate
        frame_rate: float = w.getframerate()
        # Close file
        w.close()

        # Time data
        self._time_data: np.ndarray = np.linspace(
            0, len(self._signal_data) / frame_rate, len(self._signal_data)
        )

        # Amount of time captured by `update`
        self._buffer_time: float = buffer_time
        # Chunk size and sample rate
        super().__init__(int(self._buffer_time * frame_rate), int(frame_rate))
        # Index of stream
        self._pointer: int = 0

        # Restart flag
        self._restart_flag: bool = False

    def update(self) -> np.ndarray:
        """Reads data from the input buffer.

        Returns
        -------
        signal_slice : np.ndarray
            next chunk of stream.
        """
        # Read chunk from array
        signal_slice: np.ndarray = np.array(
            self._signal_data[self._pointer : self._pointer + self._chunk]
        )
        # Move index over
        self._pointer += self._chunk
        if self._pointer > len(self._signal_data):
            # Go back to beginning
            self._pointer = 0
            self._restart_flag = True
            print("Restarting stream...")

        return signal_slice

    def close(self) -> None:
        """Closes the input stream."""
        pass

    def is_restarting(self) -> bool:
        """Returns whether the stream is restarting or not.

        Returns
        -------
        restart_flag : bool
            `True` if stream is restarting else `False`.
        """
        restart_flag = self._restart_flag
        # Reset the flag
        self._restart_flag = False

        return restart_flag
