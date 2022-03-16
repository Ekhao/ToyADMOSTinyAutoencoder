#!/usr/bin/env python
# coding: utf-8

# # TinyML Autoencoder for ToyADMOS

# Start by getting the data into the notebook. We will focus on the first case of toy car example, and use a subset of that dataset to train an autoencoder.

import tensorflow as tf
import librosa
import sklearn.metrics
import pathlib
import csv
import time
import argparse

argparser = argparse.ArgumentParser(description="A tiny autoencoder for anomaly detection in the ToyADMOS dataset")
argparser.add_argument("-n","--number_of_normal_files", type=int, help="The number of normal sound files to use", default=1800)
argparser.add_argument("-a","--number_of_anomalous_files", type=int, help="The number of anomalous sound files to use", default=400)
argparser.add_argument("-sr","--sample_rate", type=int, help="The sample rate to resample the sound files at. 0 (default) will not resample the sound file.", default=0)
argparser.add_argument("-l", "--load_model", help="Load a saved model to run inference on")
argparser.add_argument("-s", "--save_name", help="Save name for the tflite model", default="test")
args = argparser.parse_args()

NUMBER_OF_NORMAL_FILES = args.number_of_normal_files
NUMBER_OF_ANOMALOUS_FILES = args.number_of_anomalous_files
if args.sample_rate == 0:
    SAMPLE_RATE = None
else:
    SAMPLE_RATE = args.sample_rate
SAVE_NAME = args.save_name
LOAD_MODEL = args.load_model

normal_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
anomalous_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"

# Get the file paths for the sound files in the training and test path
normal_files_path = tf.io.gfile.glob(normal_path + "*ch1*.wav")
anomalous_files_path = tf.io.gfile.glob(anomalous_path + "*ch1*.wav")

# Let us reduce the amount of training and test samples for the moment
normal_files_path = tf.convert_to_tensor(normal_files_path[:NUMBER_OF_NORMAL_FILES])
anomalous_files_path = tf.convert_to_tensor(anomalous_files_path[:NUMBER_OF_ANOMALOUS_FILES])


def custom_librosa_load(audio_file):
    audio, _ = librosa.load(audio_file.numpy(), sr = SAMPLE_RATE)
    return audio

# Load the audio files
normal_audio = tf.map_fn(fn=custom_librosa_load, elems=normal_files_path, fn_output_signature=tf.float32)
anomalous_audio = tf.map_fn(fn=custom_librosa_load, elems=anomalous_files_path, fn_output_signature=tf.float32)


# Now that we have the audio loaded into memory, we create spectrograms based on the audio.
FRAME_SIZE = 2048
HOP_SIZE = 512

def apply_stft(audio_sample):
    mel_spectrogram = librosa.feature.melspectrogram(audio_sample.numpy(), sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=256)
    return librosa.power_to_db(mel_spectrogram)

normal_magnitudes = tf.map_fn(fn=apply_stft, elems=normal_audio)
anomalous_magnitudes = tf.map_fn(fn=apply_stft, elems=anomalous_audio)

# We need to add a new dimension to the datasets to be able to work with tensorflows convolutions that expect dimensions that would normally be present in an image (dataset_dimension, x, y, channels)
normal_magnitudes = normal_magnitudes[..., tf.newaxis]
anomalous_magnitudes = anomalous_magnitudes[..., tf.newaxis]

# We pad the spectrograms to multiples of 4 to make max pooling and upsampling result in the same shape
current_rows = normal_magnitudes.shape[1]
current_cols = normal_magnitudes.shape[2]

rows_to_add = abs((current_rows%4)-4)%4
cols_to_add = abs((current_cols%4)-4)%4

normal_magnitudes = tf.keras.layers.ZeroPadding2D(padding=((rows_to_add,0),(cols_to_add,0)))(normal_magnitudes)
anomalous_magnitudes = tf.keras.layers.ZeroPadding2D(padding=((rows_to_add,0),(cols_to_add,0)))(anomalous_magnitudes)

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

    training_magnitudes = shuffled_normal_magnitudes[:int(NUMBER_OF_NORMAL_FILES*0.6)]
    validation_magnitudes = shuffled_normal_magnitudes[int(NUMBER_OF_NORMAL_FILES*0.6):int(NUMBER_OF_NORMAL_FILES*0.8)]
    test_magnitudes = shuffled_normal_magnitudes[int(NUMBER_OF_NORMAL_FILES*0.8):]

    test_magnitudes = tf.concat([test_magnitudes, anomalous_magnitudes], axis=0)


    # We name the training, validation and test data according to common conventions

    x_train = training_magnitudes
    x_validate = validation_magnitudes
    x_test = test_magnitudes


    print(f"Training samples: {len(training_magnitudes)}, Validation samples: {len(validation_magnitudes)}, Test samples: {len(test_magnitudes)}")


    # ## Defining the model
    # We now define an autoencoder to take the preprocessed spectrograms as input.

    encoder = tf.keras.models.Sequential()

    encoder.add(tf.keras.layers.Conv2D(32, 3, padding="same",
                                   activation="relu", input_shape=x_train.shape[1:]))
    encoder.add(tf.keras.layers.MaxPooling2D())

    encoder.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
    encoder.add(tf.keras.layers.MaxPooling2D())

    encoder.summary()


    decoder = tf.keras.models.Sequential()

    decoder.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))
    decoder.add(tf.keras.layers.UpSampling2D())

    decoder.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))
    decoder.add(tf.keras.layers.UpSampling2D())

    decoder.add(tf.keras.layers.Conv2D(1, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))

    decoder.summary()


    conv_autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))


    conv_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())


    history = conv_autoencoder.fit(x_train, x_train, validation_data=(x_validate, x_validate), epochs=50)
    return x_test,conv_autoencoder

def load_model(normal_magnitudes, anomalous_magnitudes):
    x_test = tf.concat([normal_magnitudes, anomalous_magnitudes], axis=0)
    save_path = f"./saved_tf_models/{LOAD_MODEL}/"
    return x_test, tf.keras.models.load_model(save_path)

if LOAD_MODEL == None:
    x_test, conv_autoencoder = train_model(normal_magnitudes, anomalous_magnitudes)
else:
    x_test, conv_autoencoder = load_model(normal_magnitudes, anomalous_magnitudes)



# ### We plot an ROC and PR curve to evaluate the model and set a treshold

# First we generate a tensor containin the ground truth values of the test set and do predictions on the test set

ground_truths_normal = tf.constant(False, shape=len(x_test)-NUMBER_OF_ANOMALOUS_FILES, dtype=bool)
ground_truths_anomaly = tf.constant(True, shape=NUMBER_OF_ANOMALOUS_FILES, dtype=bool)
ground_truths = tf.concat([ground_truths_normal, ground_truths_anomaly], axis=0)


start_inference = time.time()
test_reconstructions = conv_autoencoder.predict(x_test)
inference_time = time.time()-start_inference

reshaped_x_test = tf.reshape(x_test, [x_test.shape[0], tf.math.reduce_prod(x_test.shape[1:])])
reshaped_test_reconstructions = tf.reshape(test_reconstructions, [test_reconstructions.shape[0], tf.math.reduce_prod(test_reconstructions.shape[1:])])

test_losses = tf.keras.losses.mse(reshaped_x_test, reshaped_test_reconstructions)


# We can then create the roc_curve

fpr, tpr, thr = sklearn.metrics.roc_curve(tf.cast(ground_truths, dtype=tf.float32), test_losses)


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
    reshaped_data = tf.reshape(data, [data.shape[0], tf.math.reduce_prod(data.shape[1:])])
    reshaped_reconstructions = tf.reshape(reconstructions, [reconstructions.shape[0], tf.math.reduce_prod(reconstructions.shape[1:])])
    loss = tf.keras.losses.mse(reshaped_data, reshaped_reconstructions)
    return tf.math.less(treshold, loss)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(sklearn.metrics.accuracy_score(labels, predictions)))
    print("Precision = {}".format(sklearn.metrics.precision_score(labels, predictions)))
    print("Recall = {}".format(sklearn.metrics.recall_score(labels, predictions)))


predictions = predict(conv_autoencoder, x_test, threshold)


print_stats(predictions, ground_truths)


# ## Save the model to load it using the tflite converter
# Can then also be loaded in case the training takes a long time

conv_autoencoder.save("./saved_tf_models/test/")


# ## Convert the model to a Tensorflow Lite Micro Model

tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model("./saved_tf_models/test/")

tf_lite_model = tf_lite_converter.convert()
tf_lite_model_dir = pathlib.Path("./tf_lite_models/")

tf_lite_model_file = tf_lite_model_dir/"modeltest.tflite"
model_size = tf_lite_model_file.write_bytes(tf_lite_model)

# ## Save the results in a csv file

# If the results.csv file does not exist we open it and write the header
if not pathlib.Path("results.csv").is_file():
    with open("results.csv", "w", encoding="UTF8") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["sample_rate(Hz)", "auc_score", "tf_lite_model_size(bytes)" , "inference_time(seconds)"])
        
# Then write the results of this run
with open("results.csv", "a", encoding="UTF8") as results_file:
    writer = csv.writer(results_file)
    writer.writerow([SAMPLE_RATE, round(roc_auc,6), model_size, round(inference_time,6)])




