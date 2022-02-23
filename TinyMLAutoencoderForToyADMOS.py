#!/usr/bin/env python
# coding: utf-8

# # TinyML Autoencoder for ToyADMOS

# Start by getting the data into the notebook. We will focus on the first case of toy car example, and use a subset of that dataset to train an autoencoder.

# In[1]:


import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn.metrics
import pathlib
import csv
import time
import sys


# In[2]:


# In the pure python version of the program we use command line arguments to decide number of files and sample rate.
if len(sys.argv) != 4:
    print(f"Usage: TinyMLAutoencoderForToyADMOS.py <NUMBER_OF_NORMAL_FILES> <NUMBER_OF_ANOMALOUS_FILES> <SAMPLE_RATE> \nSample rate can be set to None to preserve original sample rate of files.")
    sys.exit()
NUMBER_OF_NORMAL_FILES = int(sys.argv[1])
NUMBER_OF_ANOMALOUS_FILES = int(sys.argv[2])
if sys.argv[3] == "None":
    SAMPLE_RATE = None
else:
    SAMPLE_RATE = int(sys.argv[3])


# In[3]:


normal_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/NormalSound_IND/"
anomalous_path = "/Users/emjn/Documents/DTU/Datasets/ToyConveyor/case1/AnomalousSound_IND/"

# Get the file paths for the sound files in the training and test path
normal_files_path = tf.io.gfile.glob(normal_path + "*ch1*.wav")
anomalous_files_path = tf.io.gfile.glob(anomalous_path + "*ch1*.wav")

# Let us reduce the amount of training and test samples for the moment
normal_files_path = tf.convert_to_tensor(normal_files_path[:NUMBER_OF_NORMAL_FILES])
anomalous_files_path = tf.convert_to_tensor(anomalous_files_path[:NUMBER_OF_ANOMALOUS_FILES])


# In[4]:


# Get sample rate
audio_file = normal_files_path[0].numpy()
_, sr = librosa.load(audio_file, sr = SAMPLE_RATE)
print(f"Using sample rate: {sr}")


# In[5]:


def custom_librosa_load(audio_file):
    audio, _ = librosa.load(audio_file.numpy(), sr = SAMPLE_RATE)
    return audio


# In[6]:


# This is hopefully faster
normal_audio = tf.map_fn(fn=custom_librosa_load, elems=normal_files_path, fn_output_signature=tf.float32)
anomalous_audio = tf.map_fn(fn=custom_librosa_load, elems=anomalous_files_path, fn_output_signature=tf.float32)


# Now that we have the audio loaded into memory, we create spectrograms based on the audio.

# In[7]:


FRAME_SIZE = 2048
HOP_SIZE = 512


# In[8]:


def apply_stft(audio_sample):
    mel_spectrogram = librosa.feature.melspectrogram(audio_sample.numpy(), sr=sr, n_fft=2048, hop_length=512, n_mels=256)
    return librosa.power_to_db(mel_spectrogram)


# In[9]:


normal_magnitudes = tf.map_fn(fn=apply_stft, elems=normal_audio)
anomalous_magnitudes = tf.map_fn(fn=apply_stft, elems=anomalous_audio)


# Let us visualize a spectrogram of one of the sound files

# In[10]:

# There is no need to visualise anything during the experiments

#plt.figure(figsize=(25,10))
#librosa.display.specshow(normal_magnitudes[2].numpy(), sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
#plt.colorbar(format="%+2.f")


# ## Preprocessing input audio
# Having generated the spectrograms we now need to preprocess the them to be suitable for the network. 

# In[11]:


#normal_magnitudes.shape


# We need to add a new dimension to the datasets to be able to work with tensorflows convolutions that expect dimensions that would normally be present in an image (dataset_dimension, x, y, channels)

# In[12]:


normal_magnitudes_4D = normal_magnitudes[..., tf.newaxis]
anomalous_magnitudes_4D = anomalous_magnitudes[..., tf.newaxis]

# We pad the spectrograms to multiples of 4 to make max pooling and upsampling result in the same shape
current_rows = normal_magnitudes_4D.shape[1]
current_cols = normal_magnitudes_4D.shape[2]

rows_to_add = abs((current_rows%4)-4)%4
cols_to_add = abs((current_cols%4)-4)%4

normal_magnitudes_padded = tf.keras.layers.ZeroPadding2D(padding=((rows_to_add,0),(cols_to_add,0)))(normal_magnitudes_4D)
anomalous_magnitudes_padded = tf.keras.layers.ZeroPadding2D(padding=((rows_to_add,0),(cols_to_add,0)))(anomalous_magnitudes_4D)

#normal_magnitudes_padded.shape


# As it can be seen on the spectrogram plotted above, the current magnitudes range from about +35 to -45. It is problematic that the values include negative numbers as it will not be possible to reconstruct these using the ReLU activation function. Furthermore neural networks work the best in values ranging from 0-1.
# 
# Therefore we apply a shift and a scale to keep the magnitudes of the spectrograms inside this range.

# In[13]:


# Getting the min and max value to compute the shift and scale
min_value = tf.reduce_min(normal_magnitudes_padded)
max_value = tf.reduce_max(normal_magnitudes_padded)

# Apply shift
normal_magnitudes_shifted = tf.math.subtract(normal_magnitudes_padded, min_value)
anomalous_magnitudes_shifted = tf.math.subtract(anomalous_magnitudes_padded, min_value)

# Apply scale
normal_magnitudes_scaled = normal_magnitudes_shifted / max_value
anomalous_magnitudes_scaled = anomalous_magnitudes_shifted / max_value


# ## Let us turn the normal and abnormal data into training, validation and test data

# In[14]:


shuffled_normal_magnitudes = tf.random.shuffle(normal_magnitudes_scaled)

training_magnitudes = shuffled_normal_magnitudes[:int(NUMBER_OF_NORMAL_FILES*0.6)]
validation_magnitudes = shuffled_normal_magnitudes[int(NUMBER_OF_NORMAL_FILES*0.6):int(NUMBER_OF_NORMAL_FILES*0.8)]
test_magnitudes = shuffled_normal_magnitudes[int(NUMBER_OF_NORMAL_FILES*0.8):]

test_magnitudes = tf.concat([test_magnitudes, anomalous_magnitudes_scaled], axis=0)


# And finally we name the training, validation and test data according to common conventions

# In[15]:


x_train = training_magnitudes
x_validate = validation_magnitudes
x_test = test_magnitudes


print(f"Training samples: {len(training_magnitudes)}, Validation samples: {len(validation_magnitudes)}, Test samples: {len(test_magnitudes)}")


# ## Defining the model
# We now define an autoencoder to take the preprocessed spectrograms as input.

# In[16]:


encoder = tf.keras.models.Sequential()

encoder.add(tf.keras.layers.Conv2D(32, 3, padding="same",
                                   activation="relu", input_shape=x_train.shape[1:]))
encoder.add(tf.keras.layers.MaxPooling2D())

encoder.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
encoder.add(tf.keras.layers.MaxPooling2D())

encoder.summary()


# In[17]:


decoder = tf.keras.models.Sequential()

decoder.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))
decoder.add(tf.keras.layers.UpSampling2D())

decoder.add(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))
decoder.add(tf.keras.layers.UpSampling2D())

decoder.add(tf.keras.layers.Conv2D(1, 3, padding="same", activation="relu", input_shape=encoder.output.shape[1:]))

decoder.summary()


# In[18]:


conv_autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))


# In[19]:


conv_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())


# In[20]:


history = conv_autoencoder.fit(x_train, x_train, validation_data=(x_validate, x_validate), epochs=50)


# In[21]:


#plt.plot(history.history["loss"], label="Training Loss")
#plt.plot(history.history["val_loss"], label="Validation Loss")
#plt.legend()


# ## Take a look at the reconstructions

# In[22]:


#reconstructions = conv_autoencoder.predict(x_train)


# In[23]:


#squeezed_reconstructions = tf.squeeze(reconstructions)
#squeezed_x_train = tf.squeeze(x_train)


# To get back to the original spectrogram scale we shift and scale the reconstructed spectrograms.

# In[24]:


#scaled_reconstructions = squeezed_reconstructions * max_value
#scaled_x_train = squeezed_x_train * max_value
#
#shifted_reconstructions = tf.math.add(scaled_reconstructions, min_value)
#shifted_x_train = tf.math.add(scaled_x_train, min_value)


# ### An original spectrogram

# In[25]:


# Which spectrogram to visualize
#spec_number=8


# In[26]:


#plt.figure(figsize=(25,10))
#librosa.display.specshow(shifted_x_train[spec_number].numpy(), sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
#plt.colorbar(format="%+2.f")


# ### The reconstructed image
# There seems to be an issue with the autoencoder not generating negative numbers!

# In[27]:


#plt.figure(figsize=(25,10))
#librosa.display.specshow(shifted_reconstructions[spec_number].numpy(), sr=sr, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
#plt.colorbar(format="%+2.f")


# ## We now have to define a treshold for when to reject a sample as normal

# In[28]:


#normal_reconstructions = reconstructions
#
## Reshape the input and reconstructions to work with tensorflow loss functions
#reshaped_x_train = tf.reshape(x_train, [x_train.shape[0], tf.math.reduce_prod(x_train.shape[1:])])
#reshaped_normal_reconstructions = tf.reshape(normal_reconstructions, [normal_reconstructions.shape[0], tf.math.reduce_prod(normal_reconstructions.shape[1:])])
#
#normal_losses = tf.keras.losses.mse(reshaped_x_train, reshaped_normal_reconstructions)
#
#plt.hist(normal_losses[None,:], bins=10)
#plt.xlabel("Normal loss")
#plt.ylabel("No of examples")
#plt.show()
#
#
## Lets try to reconstruct anomalous data to see if we see higher reconstruction errors
#
## In[29]:
#
#
#anomalous_reconstructions = conv_autoencoder.predict(anomalous_magnitudes_scaled)
#
#reshaped_anomalous_magnitudes = tf.reshape(anomalous_magnitudes_scaled, [anomalous_magnitudes_scaled.shape[0], tf.math.reduce_prod(anomalous_magnitudes_scaled.shape[1:])])
#reshaped_anomalous_reconstructions = tf.reshape(anomalous_reconstructions, [anomalous_reconstructions.shape[0], tf.math.reduce_prod(anomalous_reconstructions.shape[1:])])
#
#anomalous_losses = tf.keras.losses.mse(reshaped_anomalous_magnitudes, reshaped_anomalous_reconstructions)
#
#plt.hist(anomalous_losses[None,:], bins=10)
#plt.xlabel("Anomalous loss")
#plt.ylabel("No of examples")
#plt.show()


# ### We plot an ROC and PR curve to evaluate the model and set a treshold

# First we generate a tensor containin the ground truth values of the test set and do predictions on the test set

# In[30]:


ground_truths_normal = tf.constant(False, shape=len(test_magnitudes)-NUMBER_OF_ANOMALOUS_FILES, dtype=bool)
ground_truths_anomaly = tf.constant(True, shape=NUMBER_OF_ANOMALOUS_FILES, dtype=bool)
ground_truths = tf.concat([ground_truths_normal, ground_truths_anomaly], axis=0)


# In[31]:


start_inference = time.time()
test_reconstructions = conv_autoencoder.predict(x_test)
inference_time = time.time()-start_inference

reshaped_x_test = tf.reshape(x_test, [x_test.shape[0], tf.math.reduce_prod(x_test.shape[1:])])
reshaped_test_reconstructions = tf.reshape(test_reconstructions, [test_reconstructions.shape[0], tf.math.reduce_prod(test_reconstructions.shape[1:])])

test_losses = tf.keras.losses.mse(reshaped_x_test, reshaped_test_reconstructions)


# We can then create the roc_curve

# In[32]:


fpr, tpr, thr = sklearn.metrics.roc_curve(tf.cast(ground_truths, dtype=tf.float32), test_losses)


# We calculate the optimal threshold of the ROC curve using Youden's J statistic

# In[33]:


J = tpr - fpr
index = tf.argmax(J)
threshold = thr[index]

# This threshold is one of the losses, typically one of the losses of an anomalous sound. 
# Therefore we subtract an epsilon to properly classify this in predictions. 
threshold = threshold - tf.keras.backend.epsilon()
threshold


# And finally plot the ROC curve and its AUC score

# In[34]:


roc_auc = sklearn.metrics.auc(fpr, tpr)
print("AUC score:", roc_auc)


# ## With the threshold defined, let finally try to make some predictions.

# In[37]:


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


# In[38]:


predictions = predict(conv_autoencoder, x_test, threshold)


# In[39]:


print_stats(predictions, ground_truths)


# ## Save the model for another time
# Good if training takes a long time

# In[40]:


conv_autoencoder.save("./saved_tf_models/test/")

# ## Convert the model to a Tensorflow Lite Micro Model

# In[42]:


tf_lite_converter = tf.lite.TFLiteConverter.from_saved_model("./saved_tf_models/test/")


# In the block below we run optimizations

# In[43]:


## Don't use optimizations to convert to tf lite, as we do not run inference or get auc score after the optimizations
#tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
#def representative_data_gen():
#    yield [x_train[:100]]
#tf_lite_converter.representative_dataset = representative_data_gen
#tf_lite_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


# In[44]:


tf_lite_model = tf_lite_converter.convert()
tf_lite_model_dir = pathlib.Path("/Users/emjn/Documents/DTU/PlayingWithDatasets/ToyADMOSTinyAutoencoder/tf_lite_models/")

tf_lite_model_file = tf_lite_model_dir/"modeltest.tflite"
model_size = tf_lite_model_file.write_bytes(tf_lite_model)


# ## Save the results in a csv file

# In[45]:


# If the results.csv file does not exist we open it and write the header
if not pathlib.Path("results.csv").is_file():
    with open("results.csv", "w", encoding="UTF8") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["sample_rate(Hz)", "auc_score", "tf_lite_model_size(bytes)" , "inference_time(seconds)"])
        
# Then write the results of this run
with open("results.csv", "a", encoding="UTF8") as results_file:
    writer = csv.writer(results_file)
    writer.writerow([sr, round(roc_auc,6), model_size, round(inference_time,6)])


# In[ ]:




