"""Program that can record SpikerBox data and save it as a numpy array file."""

# Standard modules
import argparse
import sys
import time
from typing import List

# External modules


# Internal modules
from data_stream import *

SAVE_FOLDER = "recordings"


def main(args: List = sys.argv):
    """Main entry point.
    Parameters
    ----------
    args : List
        unparsed command line arguments. Default value is sys.argv.
    """
    # Parse command line arguments
    parsed_args = _parse_args(args[1:])

    # Create input stream
    input_stream = SpikerStream(parsed_args.serial_port, parsed_args.buffer_time)

    start_time = time.time()
    print("Starting recording...")
    save_array = input_stream.update()
    elapsed = 0
    while elapsed < float(parsed_args.recording_time):
        save_array = np.concatenate([input_stream.update(), save_array])
        elapsed = time.time() - start_time
    time_slice = np.linspace(0, elapsed, len(save_array))
    np.save(
        f"{SAVE_FOLDER}/{parsed_args.filename[0]}.npy",
        np.vstack((save_array, time_slice)),
    )
    print(f"Recording finished in {elapsed} seconds")


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
        description="This CLI program is used to test out the SpikerStream interface."
    )

    parser.add_argument(
        "-s",
        "--s",
        "-serialport",
        "--serialport",
        nargs="?",
        default="COM1",
        help="The serial port the SpikerBox is attached to.",
        dest="serial_port",
    )

    parser.add_argument(
        "-t",
        "--t",
        "-time",
        "--time",
        nargs="?",
        default=30,
        help="The recording time in seconds.",
        dest="recording_time",
    )

    parser.add_argument(
        "filename",
        nargs=1,
        type=str,
        help="The file name of the saved array.",
    )

    parser.add_argument(
        "-b",
        "--b",
        "-buffer_time",
        "--buffer_time",
        nargs="?",
        default=1.5,
        help="The size of the buffer in seconds.",
        dest="buffer_time",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
