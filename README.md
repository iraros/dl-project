# Using Transfer Learning and Audio-Visual adaptation for Sound Classification


A final project for the course [Deep Learning](https://www30.tau.ac.il/yedion/syllabus.asp?course=0368448801) with professor [Lior Wolf](https://www.cs.tau.ac.il/~wolf/) of Tel Aviv University, Fall 2018 

This README provides explanations for the code in the project.
Further explanations on the project itself can be found in [its google doc](https://docs.google.com/document/d/1o34ps-ylUmQE6mW7BKTi74rkhj25P-hmon9d7wYMQ3k/edit?usp=sharing).


## external scripts

### retrain.py

main script from [TensorHub](https://www.tensorflow.org/hub/tutorials/image_retraining).
Contains many options:

```
usage: retrain.py [-h] [--image_dir IMAGE_DIR] [--output_graph OUTPUT_GRAPH]
                  [--intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR]
                  [--intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY]
                  [--output_labels OUTPUT_LABELS]
                  [--summaries_dir SUMMARIES_DIR]
                  [--how_many_training_steps HOW_MANY_TRAINING_STEPS]
                  [--learning_rate LEARNING_RATE]
                  [--testing_percentage TESTING_PERCENTAGE]
                  [--validation_percentage VALIDATION_PERCENTAGE]
                  [--eval_step_interval EVAL_STEP_INTERVAL]
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--test_batch_size TEST_BATCH_SIZE]
                  [--validation_batch_size VALIDATION_BATCH_SIZE]
                  [--print_misclassified_test_images]
                  [--bottleneck_dir BOTTLENECK_DIR]
                  [--final_tensor_name FINAL_TENSOR_NAME] [--flip_left_right]
                  [--random_crop RANDOM_CROP] [--random_scale RANDOM_SCALE]
                  [--random_brightness RANDOM_BRIGHTNESS]
                  [--tfhub_module TFHUB_MODULE]
                  [--saved_model_dir SAVED_MODEL_DIR]
                  [--logging_verbosity {DEBUG,INFO,WARN,ERROR,FATAL}]
```

As stated before, more information can be obtained by running ```!python retrain.py -h```.
I modified this file very little.

### label_image_script.py

Basically [This](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/label_image.py) TensorHub script.
Is the inspiration for my labeler.

Usage example from TensorHub:
```
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

## My code


### constants.py

A constants file, contains class number to name mapping.

### converter.py

Used to create visual features.
Loads all the wav files and converts them according to various specifications.
Based on [this code](https://gist.github.com/lukeinator42/917308d61b6a44afd3739c5ca73ec82b#file-spectrogram-py) by [**Lukas Grasse**](https://medium.com/@lukasgrasse).

```
# dependencies
csv
os
librosa
numpy as np
matplotlib.pyplot as plt
librosa.display
```

### evaluator.py

Used to perform multiple folds evaluation.

```
# dependencies
(local storage)
json
os
random
labeler
numpy as np
```

### labeler.py

Used to label images, i.e. to perform inference on a given model.
Inspired by the labeling script from TensorHub.

```
# dependencies
os
numpy as np
tensorflow as tf
```

### merger.py

Used to merge (or stack) plots together.
Allows fiddling with strength, order and variability of plots.

```
# dependencies
(local storage)
os
re
numpy as np
from PIL Image
utils
```

### trainer.py

Used to pre-process images to sub-folders fitting the retrain script format.
Also create fold specific training commands, that are run through Train.ipynb.

```
# dependencies
(local storage)
json
os
random
labeler
numpy as np
```

## My IPython Notebooks


### plotting trials.ipynb

Contains simple code that I used to try an plot new features from the sound files,
before implementing them in the converter.
Just on cell, when it is run - the plot appears.
Like converter.py, based on excerpts from  **Lukas Grasse**'s code.

```
# dependencies:
(local storage)
numpy as np
matplotlib.pyplot as plt
librosa.display
```

### visualize.ipynb

Facilitates labeling (by model inference) on a random image, than visualizing the image and playing the sound.
Has a cell for class inference, another for image visualizing and another for sound playing.

```
# dependencies:
(local storage)
labeler
utils
IPython.display
random
```

### train.ipynb

Runs the retrain.py TensorHub script.
Based on the instructions by [TensorFlow](https://www.tensorflow.org/hub/tutorials/image_retraining) and the instructions in the script itself, visible by running
```!python retrain.py -h``` 
Contains 2 cell, one for systemic runs (and plays an alarm at the end), another for more casual experiments.

```
# dependencies:
(local storage)
retrain
IPython.display
```

## Example data

Here is one specific representative audio file (which I personally like)
of a dog barking, and a few of its image feature representations.

### 344-3-5-0.wav

The wav sound itself. It is named by the UrbanSound8K data set conventions:
[fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:

[fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken

[classID] = a numeric identifier of the sound class (see description of classID below for further details)

[occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording

[sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence


### gray_scale_spectrogram.png

A gray scale version of the logarithmic mel scale spectrogram.

### non_log_spectrogram.png

A linear spectrogram.

### spectrogram.png

A mel scale colorful spectrogram.

### spectrogram_and_wave_form_merged.png

A spectrogram and waveform plot merged as channels of the same image,
with green background as the third channel.

### spectrogram_and_wave_form_stacked.png

A spectrogram and waveform plot stacked on top of each other.

### wave_form.png

A waveform of the audio.