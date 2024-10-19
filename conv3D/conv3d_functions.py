import os
import cv2
import numpy as np
import datetime as dt
import shutil
import pandas as pd
import random
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps

import tensorflow as tf

from keras.layers import *
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# List files and ignore .DS_Store if on a Mac
def list_files(directory):
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)

    return visible_files

# 1. Load and Preprocess Videos
def load_videos_from_folders(folder_path, img_size=(64, 64), sequence_length=30):
    # classes = os.listdir(folder_path)
    classes = list_files(folder_path)
    data, labels = [], []

    for label, activity in enumerate(classes):
        activity_folder = os.path.join(folder_path, activity)
        # for video_file in os.listdir(activity_folder):
        for video_file in list_files(activity_folder):
            video_path = os.path.join(activity_folder, video_file)
            frames = video_to_frames(video_path, img_size, sequence_length)
            if frames is not None:
                data.append(frames)
                labels.append(label)

    data = np.array(data)
    labels = to_categorical(labels, num_classes=len(classes))

    return data, labels, classes


def video_to_frames(video_path, img_size, sequence_length):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)
        if len(frames) == sequence_length:
            break
    cap.release()

    if len(frames) < sequence_length:
        return None  # Ignore short videos

    return np.array(frames)


# Build the Model
def build_3dcnn(input_shape, num_classes, conv_filters=[64, 128, 256], kernel_size=(3, 3, 3), dense_units=1024, dropout_rate=0.5, learning_rate=0.001):
    model = Sequential()

    # First Conv Layer
    model.add(Conv3D(conv_filters[0], kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Second Conv Layer
    model.add(Conv3D(conv_filters[1], kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Third Conv Layer
    model.add(Conv3D(conv_filters[2], kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 3. Run multiple iterations of training and testing, and save models
def run_multiple_iterations(train_data_folder, iterations=5, epochs_list=[10, 20]):
    # Load data
    data, labels, activity_classes = load_videos_from_folders(train_data_folder)
    input_shape = (30, 64, 64, 3)  # (sequence_length, img_size, img_size, channels)
    num_classes = len(activity_classes)

    # To store results
    all_accuracies = []
    best_accuracy = 0.0
    best_model_path = None

    # Run for multiple iterations
    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=i)

        # Build and train the model
        model = build_3dcnn(input_shape, num_classes)

        # Train model for varying epochs
        for epochs in epochs_list:
            print(f"\nTraining with {epochs} epochs...")
            # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, verbose=1)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, verbose=1)  # added after 1st run
              # Add stratification

            # Evaluate model on validation set
            val_predictions = model.predict(X_val)
            val_predicted_labels = np.argmax(val_predictions, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)

            # Calculate accuracy
            accuracy = accuracy_score(val_true_labels, val_predicted_labels)
            print(f"Validation Accuracy: {accuracy}")
            all_accuracies.append(accuracy)

            # Save the model if it's the best one so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # best_model_path = f"best_model_iter_{i+1}_epochs_{epochs}.h5"
                # model.save(best_model_path)
                # print(f"Model saved as {best_model_path} with accuracy {best_accuracy}")
               
                # Save model
                date_time_format = '%Y-%m-%d-%H-%M-%S'
                current_date_time_dt = dt.datetime.now()
                current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

                model_name = f'{current_date_time_string}-conv3d-model.keras'
                
                model.save(model_name)

                print("Done") 

    # Calculate average accuracy over all iterations
    avg_accuracy = np.mean(all_accuracies)
    print(f"\nAverage validation accuracy over {iterations} iterations: {avg_accuracy}")

    return avg_accuracy, best_model_path
    

# Define random search function
def train3d(train_data_folder, param_grid, iterations=3, random_search=False, n_combinations=10):
    # Load data
    data, labels, activity_classes = load_videos_from_folders(train_data_folder)
    input_shape = (30, 64, 64, 3)  # (sequence_length, img_size, img_size, channels)
    num_classes = len(activity_classes)

    # To store results
    best_accuracy = 0.0
    best_params = None
    best_model_path = None
    best_val_true_labels = None
    best_val_predicted_labels = None

    if random_search:
        # Randomly sample a subset of parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), values)) for values in random.sample(list(product(*param_grid.values())), n_combinations)]
    else:
        param_combinations = [param_grid]

    for param_comb in param_combinations:
        # Unpack the current parameter combination
        params = dict(param_comb)
        print(f"\nTesting with parameters: {params}")

        for i in range(iterations):
            print(f"\nIteration {i + 1}/{iterations}")
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=i)

            # Build and train the model with the current hyperparameters
            model = build_3dcnn(
                input_shape, num_classes,
                conv_filters=params['conv_filters'],
                kernel_size=params['kernel_size'],
                dense_units=params['dense_units'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )

            # Train the model for a set number of epochs
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params['epochs'], batch_size=params['batch_size'], verbose=1)

            # Evaluate model on validation set
            val_predictions = model.predict(X_val)
            val_predicted_labels = np.argmax(val_predictions, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)

            # Calculate accuracy for this iteration
            accuracy = accuracy_score(val_true_labels, val_predicted_labels)
            print(f"Iteration {i + 1} Validation Accuracy: {accuracy}")

            # Check if this is the best accuracy so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                # best_model_path = f"best_model_{params}_accuracy_{best_accuracy:.2f}.h5"
                # model.save(best_model_path)

                # Save model
                date_time_format = '%Y-%m-%d-%H-%M-%S'
                current_date_time_dt = dt.datetime.now()
                current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

                model_name = f'{current_date_time_string}-conv3d-model.keras'
                
                model.save(model_name)

                print(f"Best model saved with accuracy {best_accuracy}")

                # Store the best validation predictions and labels for confusion matrix
                best_val_true_labels = val_true_labels
                best_val_predicted_labels = val_predicted_labels

    # Return the best parameters and accuracy
    print(f"\nBest hyperparameters: {best_params} with accuracy: {best_accuracy}")

    print("Done")

    return best_params, best_accuracy, best_model_path, best_val_true_labels, best_val_predicted_labels


# Function to load saved model and evaluate on new test data
def load_and_evaluate_model(model_path, test_data_folder, img_size=(64, 64), sequence_length=30):
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the test data
    test_data, test_labels, activity_classes = load_videos_from_folders(test_data_folder, img_size, sequence_length)

    # Make predictions
    test_predictions = model.predict(test_data)

    # Convert predictions and labels to class indices
    predicted_labels = np.argmax(test_predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy}")

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activity_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return accuracy, predicted_labels, true_labels


if __name__ == "__main__":
    train3d()