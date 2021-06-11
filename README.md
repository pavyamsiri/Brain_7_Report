# Brain_7_Report

PHYS3888 Brain_7's Report code and notebook repository.

## Installation

All the libraries used are located in the requirements.txt and requirements_no_versions.txt files. They can be installed by calling pip install -r requirements.txt (there might errors with the yaml module).
If using Windows the install.ps1 script can be used to create a new conda environment to install the libraries into. The uninstall.ps1 script can then be used to delete the environment. Also if using Windows an PyInstaller exe was also packaged in the submission along with the necessary folders to be able to run the game without having to install all the python libraries (it does take a while to load).

## Space Run

All the code is located in the SpaceRun folder. It can be run with the default settings determined by the configuration files by running python `main.py`.
You play as a the white spaceship at the bottom of the screen which you can move using the left and right arrow keys. Your goal is to survive for as long as possible avoiding the red obstacles ![obstacle](SpaceRun/assets/obstacle.png) while collecting the blue energy cubes ![energy](SpaceRun/assets/energy.png).
The bottom right bar is your health and you lose the game when it reaches zero. The bottom left has a stop/go ![go](SpaceRun/assets/go.png) ![stop](SpaceRun/assets/stop.png) sign that indicates whether you are allowed to move when using the SpikerBox controls (SpikerStream, ArrayStream and WAVStream). The top left shows the frame rate and the top right shows the current score. When using SpikerBox controls a snapshot of the signal at the time of classification will be displayed on the right panel. The left panel will show the last control activated.

*Controls*:

* left/right arrow keys to move left/right. The game uses a toroidal geometry so moving left at the leftmost lane will move you to the rightmost lane and vice versa.
* F10 toggles keyboard and SpikerBox controls.
* F9 toggles display between frame rate (frames per second) and frame time (milliseconds per frame).
* F8 swaps left and right inputs meaning pressing left will move you right and vice versa.

### Configuration files

The SpaceRun application can be configured by going into the `settings` folder and changing values in the `.srconfig` files.

#### Assets

assets.srconfig should not be configured as it contains the file paths to the game assets.

#### Game

game.srconfig contains game parameters that change how the game runs.

* player:
    * hp = current health of player
    * max_hp = maximum health of player

* enemy:
    * hp = health of energy and obstacles.
    * spawn_time = seconds between energy and obstacle spawns.
    * speed = speed of the energy and obstacles.
    * score_gain = score gained when collecting energy.

#### Spiker Box

spiker_box.srconfig contains many parameters relevant to how the SpikerBox performs.

* buffer_time = size of the moving buffer in seconds
* update_factor = fraction of the moving buffer updated when data is streamed in.
* wait_time = time needed after a classification before being able to move again (excluding in the inherent `buffer_time` needed to fill the buffer)
* num_samples = number of samples the signal is downsampled to before classification
* quality_factor = quality factor of notch filter

* stream:
    * stream_type = type of input stream. SPB is the SpikerBox stream, ARR streams in numpy arrays and WAV streams in wav files.
    * stream_file = file path to file being streamed from. Required for ARR and WAV streams.
    * port_name = serial port name of the SpikerBox

* classifier:
    * model_type = MSC for modified simple classifier, USC for the unmodified simple classifier, KNN for KNN model, RFC for RandomForest classifier and SVC for C Support Vector model.
    * file_path = file paths to the above models.
    * event_threshold = number of zero-crossing events before signal is deemed as noise
    * positive_amplitude = positive amplitude minimum threshold
    * negative_amplitude = negative amplitude minimum threshold
    * spacing = maximum spacing between the peak and trough of the downsampled signal. As a fraction of the signal length.

## Training Models

The complex classifier models are trained using the `train_models.py` file. Calling python `train_models.py --help` will explain how to use the program. It requires a folder of small event snippets produced by the `waveform_snipper.py` script (also see python `waveform_snipper.py --help`).

## Waveform Snipper

Waveform snipper cuts out eye movement events from recorded data using an annotations file located in the same folder as the recording.

## Accuracy Notebook

This notebook calculates the accuracy of the models both in a streaming context and a snippet context. It uses functions defined in `streaming_accuracy.py` and `snippet_accuracy.py`. These scripts can be run on their own to calculate the accuracy of the current parameters defined in the configuration files. If the optimisation_flag is set to True then these scripts will also perform an optimisation algorithm to optimise the SpikerBox parameters for the highest accuracy (be warned that these files can theoretically run for several days as each evaluation takes at least 8 seconds).

## Benchmark Notebook

This notebook performs a series of benchmarks on the classifier models.

## Saver

The script `saver.py` was used to record SpikerBox data (see its help for more details on its usage).
