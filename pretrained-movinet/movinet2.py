import tqdm
import random
import pathlib
import itertools
import collections
import tarfile
import wget

import cv2
import numpy as np
import sklearn
import remotezip as rz

import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub

from keras import layers
from keras.models import Sequential
# from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

from movinet_functions import *

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(URL, 
                        num_classes = 10, 
                        splits = {"train": 30, "test": 20}, 
                        download_dir = download_dir)


batch_size = 8
num_frames = 8


output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)

for frames, labels in train_ds.take(10):
  print(labels)

print(f"Shape: {frames.shape}")
print(f"Label: {labels.shape}")

gru = layers.GRU(units=4, return_sequences=True, return_state=True)

inputs = tf.random.normal(shape=[1, 10, 8]) # (batch, sequence, channels)

result, state = gru(inputs) # Run it all at once

first_half, state = gru(inputs[:, :5, :])   # run the first half, and capture the state
second_half, _ = gru(inputs[:,5:, :], initial_state=state)  # Use the state to continue where you left off.

print(np.allclose(result[:, :5,:], first_half))
print(np.allclose(result[:, 5:,:], second_half))

model_id = 'a0'
resolution = 224

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# # Load pre-trained weights
# url = 'https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz'

# filename = wget.download(url)
# # tar -xvf movinet_a0_base.tar.gz

weights = f'movinet_{model_id}_base.tar.gz'

with tarfile.open(weights, "r") as this_tar:
    print("Opened tarfile")
    this_tar.extractall(path="./")
    print("All files extracted")
    
checkpoint_dir = f'./movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
# status = checkpoint.restore(checkpoint_path)
# status.assert_existing_objects_matched()


model = build_classifier(batch_size, num_frames, resolution, backbone, 10)

num_epochs = 2

loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_obj, optimizer='Adam', metrics=['accuracy'])

results = model.fit(train_ds,
                    validation_data=test_ds,
                    epochs=num_epochs,
                    validation_freq=1,
                    verbose=1)

model.evaluate(test_ds, return_dict=True)

fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, label_names, 'test')

