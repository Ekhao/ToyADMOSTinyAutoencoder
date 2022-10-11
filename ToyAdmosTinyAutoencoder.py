#!/usr/bin/env python
# coding: utf-8

# # TinyML Autoencoder for ToyADMOS

# Start by getting the data into the notebook. We will focus on the first case of toy car example, and use a subset of that dataset to train an autoencoder.

import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import sklearn.metrics
import pathlib
import csv
import time
import argparse
from joblib import Parallel, delayed

import configuration

argparser = argparse.ArgumentParser(
    description="A tiny convolutional autoencoder for anomaly detection in the ToyADMOS dataset")
argparser.add_argument("-n", "--number_of_normal_files", type=int,
                       help="The number of normal sound files to use", default=1800)
argparser.add_argument("-a", "--number_of_anomalous_files", type=int,
                       help="The number of anomalous sound files to use", default=400)
argparser.add_argument("-sr", "--sample_rate", type=int,
                       help="The sample rate to resample the sound files at. 0 will not resample the sound file.", default=24000)
argparser.add_argument(
    "-dt", "--data_type", help="The data type to load the audio to be processed as. Supported combinations of data type and bit width are float32, float64, int16 and int32", choices=["float32", "float64", "int16", "int32"], default="float32")
argparser.add_argument("-l", "--load_model",
                       help="Load a saved model to run inference on")
argparser.add_argument("-s", "--save_name",
                       help="Save name for the tflite model", default="test")
args = argparser.parse_args()

if args.data_type == "float32":
    args.data_type = np.float32
elif args.data_type == "float64":
    args.data_type = np.float64
elif args.data_type == "int16":
    args.data_type = np.int16
elif args.data_type == "int32":
    args.data_type = np.int32

# Get the file paths for the sound files in the training and test path
normal_files_path = tf.io.gfile.glob(
    configuration.PATH_TO_NORMAL_FILES + "*ch1*.wav")
anomalous_files_path = tf.io.gfile.glob(
    configuration.PATH_TO_ANOMALOUS_FILES + "*ch1*.wav")

# Reduce the amount of files for the experiment to what has been given at the call of the experiment
normal_files_path = normal_files_path[:args.number_of_normal_files]
anomalous_files_path = anomalous_files_path[:args.number_of_anomalous_files]

if args.sample_rate == 0:
    args.sample_rate = librosa.get_samplerate(normal_files_path[0])


# A function to load the audio files without their sample rate
def load_sound_without_sample_rate(file):
    audio, sample_rate = sf.read(
        file, dtype=args.data_type)
    # Librosa uses sound files in a transposed shape of soundfile. As we will use librosa further on we thus transpose the loaded audio. https://librosa.org/doc/main/ioformats.html#ioformats. Since we only use one channel for the sound this is actually not needed.
    audio = audio.T

    audio = librosa.to_mono(audio)

    # TODO: This is done as librosa only allows its functions to receive floating point arrays. It is not the prettiest at all.
    if args.data_type in (np.int32, np.int16):
        if not np.array_equal(audio.astype(np.float64).astype(args.data_type), audio):
            raise AssertionError(
                "Conversion from int to float for Librosa resulted in inaccuracies.")
        audio = audio.astype(np.float64)
    if args.sample_rate != None:
        audio = librosa.resample(
            y=audio, orig_sr=sample_rate, target_sr=args.sample_rate)
    return audio


# Load the audio files in parallel
print(f"Loading dataset...")
normal_audio = Parallel(
    n_jobs=-1)(delayed(load_sound_without_sample_rate)(file) for file in normal_files_path)
anomalous_audio = Parallel(
    n_jobs=-1)(delayed(load_sound_without_sample_rate)(file) for file in anomalous_files_path)


def create_mel_spectrogram(audio_sample):
    return librosa.power_to_db(librosa.feature.melspectrogram(
        y=audio_sample, sr=args.sample_rate, n_fft=configuration.FRAME_SIZE, hop_length=configuration.HOP_SIZE, n_mels=configuration.NUMBER_MEL_FILTER_BANKS))


# Now that we have the audio loaded into memory, we create spectrograms based on the audio.
print(f"Converting waveforms to mel spectrograms...")
normal_magnitudes = Parallel(
    n_jobs=-1)(delayed(create_mel_spectrogram)(audio) for audio in normal_audio)
anomalous_magnitudes = Parallel(
    n_jobs=-1)(delayed(create_mel_spectrogram)(audio) for audio in anomalous_audio)

# We need to add a new dimension to the datasets to be able to work with tensorflows convolutions that expect dimensions that would normally be present in an image (dataset_dimension, x, y, channels)
normal_magnitudes = tf.convert_to_tensor(normal_magnitudes)[..., tf.newaxis]
anomalous_magnitudes = tf.convert_to_tensor(
    anomalous_magnitudes)[..., tf.newaxis]

# We pad the spectrograms to multiples of 4 to make max pooling and upsampling result in the same shape
current_rows = normal_magnitudes.shape[1]
current_cols = normal_magnitudes.shape[2]

rows_to_add = abs((current_rows % 4)-4) % 4
cols_to_add = abs((current_cols % 4)-4) % 4

normal_magnitudes = tf.keras.layers.ZeroPadding2D(
    padding=((rows_to_add, 0), (cols_to_add, 0)))(normal_magnitudes)
anomalous_magnitudes = tf.keras.layers.ZeroPadding2D(
    padding=((rows_to_add, 0), (cols_to_add, 0)))(anomalous_magnitudes)

# The current magnitudes range from about +35 to -45. It is problematic that the values include negative numbers as it will not be possible to reconstruct these using the ReLU activation function. Furthermore neural networks work the best in values ranging from 0-1.
# Therefore we apply a shift and a scale to keep the magnitudes of the spectrograms inside this range.

# Getting the min and max value to compute the shift and scale
min_value = tf.reduce_min(normal_magnitudes)
max_value = tf.reduce_max(normal_magnitudes)

# Apply shift
normal_magnitudes = tf.math.subtract(normal_magnitudes, min_value)
anomalous_magnitudes = tf.math.subtract(anomalous_magnitudes, min_value)

# Apply scale
normal_magnitudes = normal_magnitudes / max_value
anomalous_magnitudes = anomalous_magnitudes / max_value


# We now either train the autoencoder model or load one from disk

def train_model(normal_magnitudes, anomalous_magnitudes):
    shuffled_normal_magnitudes = tf.random.shuffle(normal_magnitudes)

    training_magnitudes = shuffled_normal_magnitudes[:int(
        args.number_of_normal_files*0.6)]
    validation_magnitudes = shuffled_normal_magnitudes[int(
        args.number_of_normal_files*0.6):int(args.number_of_normal_files*0.8)]
    test_magnitudes = shuffled_normal_magnitudes[int(
        args.number_of_normal_files*0.8):]

    test_magnitudes = tf.concat(
        [test_magnitudes, anomalous_magnitudes], axis=0)

    # We name the training, validation and test data according to common conventions

    x_train = training_magnitudes
    x_validate = validation_magnitudes
    x_test = test_magnitudes

    print(
        f"Training samples: {len(training_magnitudes)}, Validation samples: {len(validation_magnitudes)}, Test samples: {len(test_magnitudes)}")

    # ## Defining the model
    # We now define an autoencoder to take the preprocessed spectrograms as input.

    encoder = tf.keras.models.Sequential()

    encoder.add(tf.keras.layers.Conv2D(32, 3, padding="same",
                                       activation="relu", input_shape=x_train.shape[1:]))
    encoder.add(tf.keras.layers.MaxPooling2D())

    encoder.add(tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"))
    encoder.add(tf.keras.layers.MaxPooling2D())

    encoder.summary()

    decoder = tf.keras.models.Sequential()

    decoder.add(tf.keras.layers.Conv2D(32, 3, padding="same",
                activation="relu", input_shape=encoder.output.shape[1:]))
    decoder.add(tf.keras.layers.UpSampling2D())

    decoder.add(tf.keras.layers.Conv2D(16, 3, padding="same",
                activation="relu", input_shape=encoder.output.shape[1:]))
    decoder.add(tf.keras.layers.UpSampling2D())

    decoder.add(tf.keras.layers.Conv2D(1, 3, padding="same",
                activation="relu", input_shape=encoder.output.shape[1:]))

    decoder.summary()

    conv_autoencoder = tf.keras.Model(
        inputs=encoder.input, outputs=decoder(encoder.outputs))

    conv_autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

    history = conv_autoencoder.fit(
        x_train, x_train, validation_data=(x_validate, x_validate), epochs=configuration.NUMBER_OF_EPOCHS)
    return x_test, conv_autoencoder


def load_model(normal_magnitudes, anomalous_magnitudes):
    x_test = tf.concat([normal_magnitudes, anomalous_magnitudes], axis=0)
    save_path = f"./saved_tf_models/{args.load_model}/"
    return x_test, tf.keras.models.load_model(save_path)


if args.load_model == None:
    x_test, conv_autoencoder = train_model(
        normal_magnitudes, anomalous_magnitudes)
else:
    x_test, conv_autoencoder = load_model(
        normal_magnitudes, anomalous_magnitudes)


# ### We plot an ROC and PR curve to evaluate the model and set a treshold

# First we generate a tensor containin the ground truth values of the test set and do predictions on the test set

ground_truths_normal = tf.constant(False, shape=len(
    x_test)-args.number_of_anomalous_files, dtype=bool)
ground_truths_anomaly = tf.constant(
    True, shape=args.number_of_anomalous_files, dtype=bool)
ground_truths = tf.concat(
    [ground_truths_normal, ground_truths_anomaly], axis=0)


start_inference = time.time()
test_reconstructions = conv_autoencoder.predict(x_test)
inference_time = time.time()-start_inference

reshaped_x_test = tf.reshape(
    x_test, [x_test.shape[0], tf.math.reduce_prod(x_test.shape[1:])])
reshaped_test_reconstructions = tf.reshape(test_reconstructions, [
    test_reconstructions.shape[0], tf.math.reduce_prod(test_reconstructions.shape[1:])])

test_losses = tf.keras.losses.mse(
    reshaped_x_test, reshaped_test_reconstructions)


# We can then create the roc_curve

fpr, tpr, thr = sklearn.metrics.roc_curve(
    tf.cast(ground_truths, dtype=tf.float32), test_losses)


# We calculate the optimal threshold of the ROC curve using Youden's J statistic

J = tpr - fpr
index = tf.argmax(J)
threshold = thr[index]

# This threshold is one of the losses, typically one of the losses of an anomalous sound.
# Therefore we subtract an epsilon to properly classify this in predictions.
threshold = threshold - tf.keras.backend.epsilon()

# And finally plot the ROC curve and its AUC score

roc_auc = sklearn.metrics.auc(fpr, tpr)
print("AUC score:", roc_auc)


# ## With the threshold defined, let finally try to make some predictions.

def predict(model, data, treshold):
    reconstructions = model.predict(data)
    reshaped_data = tf.reshape(
        data, [data.shape[0], tf.math.reduce_prod(data.shape[1:])])
    reshaped_reconstructions = tf.reshape(reconstructions, [
        reconstructions.shape[0], tf.math.reduce_prod(reconstructions.shape[1:])])
    loss = tf.keras.losses.mse(reshaped_data, reshaped_reconstructions)
    return tf.math.less(treshold, loss)


def print_stats(predictions, labels):
    print("Accuracy = {}".format(
        sklearn.metrics.accuracy_score(labels, predictions)))
    print("Precision = {}".format(
        sklearn.metrics.precision_score(labels, predictions)))
    print("Recall = {}".format(sklearn.metrics.recall_score(labels, predictions)))


predictions = predict(conv_autoencoder, x_test, threshold)


print_stats(predictions, ground_truths)


# ## Save the model to load it using the tflite converter
# Can then also be loaded in case the training takes a long time
if args.load_model == None:
    conv_autoencoder.save(f"./saved_tf_models/{args.save_name}/")
    # Convert the model to a Tensorflow Lite Micro Model
    tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model(
        f"./saved_tf_models/{args.save_name}/")
else:
    tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model(
        f"./saved_tf_models/{args.load_model}/")

tf_lite_model = tf_lite_converter.convert()
tf_lite_model_dir = pathlib.Path("./tf_lite_models/")

tf_lite_model_file = tf_lite_model_dir/f"{args.save_name}.tflite"
model_size = tf_lite_model_file.write_bytes(tf_lite_model)

# ## Save the results in a csv file

# If the results.csv file does not exist we open it and write the header
if not pathlib.Path("results.csv").is_file():
    with open("results.csv", "w", encoding="UTF8") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["sample_rate(Hz)", "auc_score",
                        "tf_lite_model_size(bytes)", "inference_time(seconds)"])

# Then write the results of this run
with open("results.csv", "a", encoding="UTF8") as results_file:
    writer = csv.writer(results_file)
    writer.writerow([args.sample_rate, round(roc_auc, 6),
                    model_size, round(inference_time, 6)])
