# Standard modules
import argparse
import os

# Hide pygame import message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import sys
from typing import List, Optional
import yaml

# External modules
from serial.serialutil import SerialException

# Local modules
from aliases import FilePath, PortName
from data_stream import StreamException, StreamType
from game import GameArgs, SpaceRun
from models import (
    Catch22Model,
    ModelBase,
    ModelException,
    ModelType,
    ModifiedModel,
    SimpleModel,
)
from spikerbox import SpikerBox


def main(args: List[str] = sys.argv[1:]):
    """Main game function.
    Handles the parsing of program arguments and configuration files to initialise the game and run the game loop.

    Parameters
    ----------
    args : List[str]
        a list of program arguments. Default is the command line arguments.
    """
    # Parse program arguments
    parsed_args: argparse.Namespace = _parse_args(args)
    # Load config files
    game_args: GameArgs = _load_game_config(parsed_args.game_config_path)
    spikerbox: SpikerBox = _load_stream_config(
        parsed_args.stream_type,
        parsed_args.model_type,
        parsed_args.buffer_time,
        parsed_args.cport,
        parsed_args.stream_file,
        parsed_args.spb_config_path,
    )
    keyboard: bool = spikerbox is None

    # Initialise the game
    game: SpaceRun = SpaceRun(
        game_args,
        parsed_args.assets_path,
        spikerbox,
        keyboard=keyboard,
    )

    # Game loop
    while True:
        game.update()
        game.draw()


def _load_game_config(config_path: FilePath) -> GameArgs:
    """Loads the game parameters using a configuration file.

    Parameters
    ----------
    config_path : FilePath
        the file path to the game configuration file.

    Returns
    -------
    game_args : GameArgs
        the arguments used to initialise the game.
    """
    # Check that the file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            str(f"The game config path is invalid! Path: {config_path}")
        )

    # Open and parse file
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        game_args = GameArgs(
            player_hp=config_data["player"]["hp"],
            player_max_hp=config_data["player"]["max_hp"],
            enemy_hp=config_data["enemy"]["hp"],
            enemy_spawn_time=config_data["enemy"]["spawn_time"],
            enemy_speed=config_data["enemy"]["speed"],
            enemy_score_gain=config_data["enemy"]["score_gain"],
        )
        return game_args


def _load_stream_config(
    stream_type: StreamType,
    model_type: ModelType,
    buffer_time: Optional[float],
    cport: Optional[PortName],
    stream_file: Optional[FilePath],
    config_path: FilePath,
) -> Optional[SpikerBox]:
    """Can create a SpikerBox object if the stream and model type are valid using program arguments and a configuration file.

    Parameters
    ----------
    stream_type: StreamType
        the type of InputStream.
    model_type: ModelType
        the type of classifier model used.
    buffer_time: Optional[float]
        an optional override of the buffer time of the SpikerBox moving buffer.
    cport: Optional[PortName]
        an optional override of the serial port of the SpikerBox.
    stream_file: Optional[FilePath]
        the file to stream data from if using an ArrayStream or WAVStream.
    config_path: FilePath
        the file path to the SpikerBox configuration file.

    Returns
    -------
    spikerbox : Optional[SpikerBox]
        a SpikerBox configured using the given configuration file and program arguments will be returned if there are no errors
        and the stream type is not NIL. Otherwise `None` is returned.
    """
    # Check that the file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"The stream config path is invalid! Path: {config_path}"
        )

    # No stream
    if stream_type == StreamType.NIL:
        return None
    # No model
    elif model_type == ModelType.NIL:
        print("No classifier model given!")
        print("Can only use keyboard controls!\n")
        return None

    # Read stream config file
    with open(config_path, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        # Overwrite configuration file settings with command line arguments
        if buffer_time is not None:
            config_data["buffer_time"] = buffer_time

        if cport is not None:
            config_data["stream"]["spiker_stream"]["port_name"] = cport

        if stream_type == StreamType.DEFAULT:
            stream_type = StreamType(config_data["stream"]["stream_type"])

        if model_type == ModelType.DEFAULT:
            model_type = ModelType(config_data["classifier"]["model_type"])

        if stream_file is not None:
            config_data["stream"]["stream_file"] = stream_file
        stream_file = config_data["stream"]["stream_file"]

        # Initialise model
        if (
            model_type == ModelType.KNN
            or model_type == ModelType.RFC
            or model_type == ModelType.SVC
        ):
            model_file = config_data["classifier"][str(model_type)]["file_path"]
            model: ModelBase = Catch22Model.create_from_file(model_file)
        elif model_type == ModelType.MSC:
            model: ModelBase = ModifiedModel(
                event_threshold=config_data["classifier"][str(ModelType.MSC)][
                    "event_threshold"
                ],
                positive_amplitude=config_data["classifier"][str(ModelType.MSC)][
                    "positive_amplitude"
                ],
                negative_amplitude=config_data["classifier"][str(ModelType.MSC)][
                    "negative_amplitude"
                ],
                spacing=config_data["classifier"][str(ModelType.MSC)]["spacing"],
            )
        elif model_type == ModelType.USC:
            model: ModelBase = SimpleModel(
                event_threshold=config_data["classifier"][str(ModelType.USC)][
                    "event_threshold"
                ],
            )
        else:
            raise ModelException("Missing a model type branch!")

        # Potentially override the default serial port
        cport = (
            config_data["stream"]["spiker_stream"]["port_name"]
            if stream_type == StreamType.SPB
            else None
        )

        # Bundle SpikerBox initialisation arguments
        try:
            return SpikerBox(
                model=model,
                buffer_time=config_data["buffer_time"],
                update_factor=config_data["update_factor"],
                wait_time=config_data["wait_time"],
                num_samples=config_data["num_samples"],
                quality_factor=config_data["quality_factor"],
                stream_type=stream_type,
                stream_file=stream_file,
                cport=cport,
            )
        except (SerialException, StreamException) as e:
            print(f"{e}\nCan only use keyboard controls!\n")
            return None


def _parse_args(args: List) -> argparse.Namespace:
    """Parses program arguments.

    Parameters
    ----------
    args : List
        unparsed program arguments

    Returns
    -------
    args : argparse.Namespace
        parsed program arguments
    """

    parser = argparse.ArgumentParser(
        description="""SpaceRun is a cross between Temple Run and Space Invaders.
        The goal is to collect energy while avoiding obstacles and obtain the highest score. It can be played using eye movements by
        connecting to a SpikerBox."""
    )

    # InputStream type
    parser.add_argument(
        "-s",
        "--s",
        "-stream",
        "--stream",
        type=StreamType,
        default=StreamType.DEFAULT,
        choices=list(StreamType),
        help="Type of stream to use.",
        dest="stream_type",
    )

    # Name of serial port
    parser.add_argument(
        "-p",
        "--p",
        "-port",
        "--port",
        type=PortName,
        nargs="?",
        default=None,
        help="The serial port of the SpikerBox.",
        dest="cport",
    )

    # Stream file
    parser.add_argument(
        "-f",
        "--f",
        "-file",
        "--file",
        type=FilePath,
        nargs="?",
        default=None,
        help="File to stream from.",
        dest="stream_file",
    )

    # Model type
    parser.add_argument(
        "-m",
        "--m",
        "-model",
        "--model",
        type=ModelType,
        default=ModelType.DEFAULT,
        choices=list(ModelType),
        help="The type of model to train.",
        dest="model_type",
    )

    # Amount of time captured by the buffer
    parser.add_argument(
        "-b",
        "--b",
        "-buffer_time",
        "--buffer_time",
        type=float,
        nargs="?",
        default=None,
        help="The size of the buffer in seconds.",
        dest="buffer_time",
    )

    # Path to SpikerBox config file
    parser.add_argument(
        "-sc",
        "--sc",
        "-spb_config",
        "--spb_config",
        type=FilePath,
        default="settings/spiker_box.srconfig",
        help="SpikerBox settings.",
        dest="spb_config_path",
    )

    # Path to Game config file
    parser.add_argument(
        "-gc",
        "--gc",
        "-game_config",
        "--game_config",
        type=FilePath,
        default="settings/game.srconfig",
        help="Game settings.",
        dest="game_config_path",
    )

    # Path to Assets config file
    parser.add_argument(
        "-ac",
        "--ac",
        "-assets_config",
        "--assets_config",
        type=FilePath,
        default="settings/assets.srconfig",
        help="Assets file paths.",
        dest="assets_path",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
