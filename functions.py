"""

Functions to train a model for human activity recognition

"""

import os
import datetime as dt
from collections import deque
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plts

import cv2

import tensorflow as tf

from keras.layers import *
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras.models import load_model

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score

# =================================
# Generic
# =================================

# List files and ignore .DS_Store if on a Mac
def list_files(directory):
    visible_files = []
    for file in os.listdir(directory):
        if not file.startswith('.'):
            visible_files.append(file)

    return visible_files


# =================================
# Conv2D
# =================================

def setup(seed_constant):
    # seed_constant = 23
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)


def get_classes(directory):
    # Get Names of all classes
    all_classes_names = list_files(directory)

    return all_classes_names


def pre_process(directory):

    classes_list = []

    labels = list_files(directory)

    for l in labels:
        classes_list.append(l)

    model_output_size = len(classes_list)

    return classes_list, model_output_size


def frames_extraction(video_path, image_height, image_width):
    # Empty List declared to store video frames
    frames_list = []

    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file
        success, frame = video_reader.read()

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Closing the VideoCapture object and releasing all resources.
    video_reader.release()

    # returning the frames list
    return frames_list


def create_dataset(directory, classes_list, image_height, image_width, max_images_per_class):

    # Declaring Empty Lists to store the features and labels values.
    temp_features = []
    features = []
    labels = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')

        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(directory, class_name))

        # Iterating through all the files present in the files list
        for file_name in files_list:
            if not file_name.startswith('.'):

                # Construct the complete video path
                video_file_path = os.path.join(directory, class_name, file_name)

                # Calling the frame_extraction method for every video file path
                frames = frames_extraction(video_file_path, image_height, image_width)

                # Appending the frames to a temporary list.
                temp_features.extend(frames)

        # Adding randomly selected frames to the features list, use random.choice()
        features.extend(random.choice(temp_features) for _ in range(max_images_per_class))

        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)

        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels


def make_features_labels(directory, classes_list, image_height, image_width, max_images_per_class):
    features, labels = create_dataset(directory, classes_list, image_height, image_width, max_images_per_class)
    return features, labels


def create_model(image_height, image_width, model_output_size):
    # Sequential model
    model = Sequential()

    # Model Architecture
    model.add(Conv2D(filters = 64, 
                     kernel_size = (3, 3), 
                     activation = 'relu', 
                     input_shape = (image_height, image_width, 3)))
    
    # Consider changing evaluation metric from micro avg to something else
        # Maybe recall, optimize for different metrics
        # Try freezing pretrained model at early or later layers, may train faster
    
    model.add(Conv2D(filters = 64, 
                     kernel_size = (3, 3), 
                     activation = 'relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, 
                    activation = 'softmax'))

    model.summary()

    return model


def train_model(model, features_train, labels_train, features_test, labels_test, epochs):

    # Adding loss, optimizer and metrics values to the model.
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])

    # Start Training
    history = model.fit(x = features_train, 
                                       y = labels_train, 
                                       epochs = epochs, 
                                       batch_size = 4, 
                                       shuffle = True, 
                                       verbose=1, 
                                       validation_split = 0.2)
    # Add stratification

    # Evaluate trained model
    model_evaluation_history = model.evaluate(features_test, labels_test)

    # Save model
    date_time_format = '%Y-%m-%d-%H-%M-%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    model_name = f'{current_date_time_string}-model.keras'
    
    model.save(model_name)

    print("Done") 


def predict_avg(directory, model, output_size, num_frames, image_height, image_width, classes, webcam=False):

    # Initialize the Numpy array which will store Prediction Probabilities
    predicted_labels_probabilities_np = np.zeros((num_frames, output_size), dtype = float)

    if webcam == True:
        # Open webcam
        video_reader = cv2.VideoCapture(0)
    else:
        video_reader = cv2.VideoCapture(directory)

    # Get The Total Frames present in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate The Number of Frames to skip Before reading a frame
    skip_frames_window = video_frames_count // num_frames

    top_predictions = []

    for frame_counter in range(num_frames):

        # Set Frame Position
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read The Frame
        _ , frame = video_reader.read()

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Pass the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Append predicted label probabilities to the deque object
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

    # Calculate Average of Predicted Labels Probabilities Column Wise
    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

    # Sort the Averaged Predicted Labels Probabilities
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    # Iterate Over All Averaged Predicted Label Probabilities
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:

        # Access The Class Name using predicted label.
        predicted_class_name = classes[predicted_label]

        # Access The Averaged Probability using predicted label.
        predicted_probability = predicted_labels_probabilities_averaged[predicted_label]
        predicted_probability = round(predicted_probability, 2)

        top_predictions.append([predicted_class_name, predicted_probability])

        # print(f"CLASS NAME: {predicted_class_name}   AVERAGED PROBABILITY: {predicted_probability}")

    # Close the VideoCapture Object and releasing all resources held by it.
    video_reader.release()

    return top_predictions


def predict_avg_stream(model, path, classes, window, image_height, image_width, print_pred=False):
    # Initialize a Deque Object with a fixed size; used to implement moving/rolling average
    predicted_labels_probabilities_deque = deque(maxlen = window)

    # Open webcam
    video_reader = cv2.VideoCapture(0)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes[predicted_label]
        
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show video player
        cv2.imshow('Predicted Frames', frame)

        # Writing The Frame, this saves the video
        video_writer.write(frame)    

        if print_pred == True:
            print('Predicted class name', predicted_class_name)

        # Press 'q' to exit the live feed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Close the VideoCapture and VideoWriter objects and release all resources held by them. 
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()


def predict_all(test_path, model, num_frames, image_height, image_width, classes, output_size, webcam):
    # train_path = f'{path}/downloads/selected_features'
    # test_path = f'{path}/downloads/ignored_features'
    train_directory = list_files(test_path)

    all_results = [] 

    dir_len = len(train_directory)
    sub_dir_len = 0
    dir_count = 1
    sub_dir_count = 1

    for directory in train_directory:

        vid_path = f'{test_path}/{directory}'

        dir_files = list_files(vid_path)
        sub_dir_len = len(dir_files)

        print(" ")
        print("=========================================")
        print(f'Class: {directory}')
        print("=========================================")

        if sub_dir_len != 0:
            for video in dir_files:
                print(" ")
                print(f'Folder: {dir_count}/{dir_len}  |  File: {sub_dir_count}/{sub_dir_len}')
                print(" ")

                input_path = f'{vid_path}/{video}'

                # Make avg prediction for each video
                p_class = predict_avg(input_path, model, output_size, num_frames, image_height, image_width, classes, webcam)
                result = [directory, p_class]
                # all_results.append(result)

                # Get true labels and predictions
                true_class = result[0]
                pred = result[1]

                for p in pred:
                    all_results.append({"true_class": true_class, 
                                "predicted_class": p[0], 
                                "predicted_value": p[1]})

                # print(result)

                sub_dir_count += 1

            sub_dir_count = 0
            dir_count += 1

        else:
            print("No video found")

    print("Done")

    return all_results


# =================================
# Plotting
# =================================

def plot_classes(directory, all_classes_names):
    # Create a Matplotlib figure
    plt.figure(figsize = (30, 30))

    # Generate a random sample of images each time the cell runs
    random_range = random.sample(range(len(all_classes_names)), 12)

    # Iterate through all the random samples
    for counter, random_index in enumerate(random_range, 1):

        # Get Class Name using Random Index
        selected_class_Name = all_classes_names[random_index]

        # Get a list of all the video files present in a Class Directory
        video_files_names_list = list_files(f'{directory}/{selected_class_Name}')

        # Randomly selecg a video file
        selected_video_file_name = random.choice(video_files_names_list)

        # Read the Video File Using the Video Capture
        video_reader = cv2.VideoCapture(f'{directory}/{selected_class_Name}/{selected_video_file_name}')

        # Read The First Frame of the Video File
        _, bgr_frame = video_reader.read()

        # Close the VideoCapture object and releasing all resources.
        video_reader.release()

        # Convert the BGR Frame to RGB Frame
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Add The Class Name Text on top of the Video Frame.
        cv2.putText(rgb_frame, selected_class_Name, (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        # Assign the Frame to a specific position of a subplot
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')


def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation']) 

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()


# Function to plot confusion matrix. This prevents notebooks from printing the plot twice.
def plot_confusion(t_class, p_class, title, cmap='turbo', **kwargs):

    # Define plot design
    title = title
    title_size = 'xx-large'
    label_size = 'large'
    tick_size = 'small'
    colors = cmap
    padding = 14

    if 'display_labels' in kwargs:
        d_labels = kwargs.get("display_labels")

    fig, ax = plt.subplots(figsize=(8,6))

    # plt.suptitle(title, fontsize = title_size)
    plt.title(title, fontsize = title_size, pad=padding * 1.25)
    plt.xticks(fontsize = tick_size)
    plt.yticks(fontsize = tick_size)
    plt.ylabel("True label", fontsize = label_size, labelpad=padding)
    plt.xlabel("Predicted label", fontsize = label_size, labelpad=padding)
    plt.subplots_adjust(bottom=0.35)

    if 'display_labels' in kwargs:
        d_labels = kwargs.get("display_labels")
        cm = ConfusionMatrixDisplay.from_predictions(t_class, p_class, cmap=colors, display_labels=d_labels)
    else:
        cm = ConfusionMatrixDisplay.from_predictions(t_class, p_class, cmap=colors)

    cm.plot(ax=ax, 
            xticks_rotation='vertical', 
            cmap=colors)
    
    plt.close()

    return fig


def get_data(file_path, top = True):
    # Import CSV of all predictions from a training run
    df = pd.read_csv(file_path)

    # Rename columns: 
    df = df.rename(columns={"true_class": "True labels", 
                            "predicted_class": "Predicted labels",
                            "predicted_value": "Predicted value"
                            })

    if top:
        # Filter data by top predictions
        top_predictions = df.iloc[::12, :].reset_index(drop=True)

        return top_predictions
    
    else:
        return df


# Make a confusion matrix from CSV data
def make_matrix_csv(file_path, plot_title, show_top=False):

    dataframe = get_data(file_path, top = show_top)

    # Select true and predicted labels
    true_classes = dataframe['True labels']
    predicted_classes = dataframe['Predicted labels']

    # Make classification report
    report = classification_report(true_classes, predicted_classes)
    print(report)
    
    # Make the plot
    confusion_matrix = plot_confusion(t_class = true_classes, 
                                      p_class = predicted_classes, 
                                      title = plot_title)
    
    return report, confusion_matrix


# =================================
# Conv3D Functions
# =================================

# Load and Preprocess Videos
def load_videos_from_folders(folder_path, img_size=(64, 64), sequence_length=30):
    # classes = os.listdir(folder_path)
    classes = list_files(folder_path)
    data, labels = [], []

    for label, activity in enumerate(classes):
        activity_folder = os.path.join(folder_path, activity)
        # for video_file in os.listdir(activity_folder):
        for video_file in list_files(activity_folder):
            video_path = os.path.join(activity_folder, video_file)
            print(f'Path: {video_path}')
            frames = video_to_frames(video_path, img_size, sequence_length)
            if frames is not None:
                data.append(frames)
                labels.append(label)

    data = np.array(data)
    labels = to_categorical(labels, num_classes=len(classes))

    return data, labels, classes


# Convert videos to frames
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


# Process frames from the webcam
def process_live_frames(img_size=(64, 64), sequence_length=30):
    cap = cv2.VideoCapture(0) 
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match the model input size
        resized_frame = cv2.resize(frame, img_size)
        frames.append(resized_frame)

        # Display the frame
        cv2.imshow('Live Video Feed', frame)

        # If we have enough frames for a sequence, yield them for prediction
        if len(frames) == sequence_length:
            yield np.array(frames)  # Return sequence as numpy array
            frames = []  # Reset for the next sequence

        # Press 'q' to exit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Predict activity on live frames
def predict_live_activity(model, activity_classes, img_size=(64, 64), sequence_length=30):
    for frames in process_live_frames(img_size, sequence_length):
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        prediction = model.predict(frames)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_activity = activity_classes[predicted_label]

        print(f"Predicted Activity: {predicted_activity}")


# Predict activity on live frames
def predict_3D(test_path, model, activity_classes, img_size=(64, 64), sequence_length=30):
    train_directory = list_files(test_path)
    all_results = [] 

    dir_len = len(train_directory)
    sub_dir_len = 0
    dir_count = 1
    sub_dir_count = 1

    for directory in train_directory:

        vid_path = f'{test_path}/{directory}'

        dir_files = list_files(vid_path)
        sub_dir_len = len(dir_files)

        print(" ")
        print("=========================================")
        print(f'Class: {directory}')
        print("=========================================")

        if sub_dir_len != 0:
            for video in dir_files:
                print(" ")
                print(f'Folder: {dir_count}/{dir_len}  |  File: {sub_dir_count}/{sub_dir_len}')
                print(" ")

                input_path = f'{vid_path}/{video}'

                # Make avg prediction for each video

            for frames in video_to_frames(input_path, img_size, sequence_length):
                # !!--------> Errors here <--------!!
                # Invalid input shape for input Tensor("data:0", shape=(1, 64, 64, 3), dtype=float32). 
                # Expected shape (None, 30, 64, 64, 3), but input has incompatible shape (1, 64, 64, 3)
                frames = np.expand_dims(frames, axis=0)  # Add batch dimension
                prediction = model.predict(frames)
                predicted_label = np.argmax(prediction, axis=1)[0]
                predicted_value= np.argmax(prediction, axis=1)[1]
                predicted_activity = activity_classes[predicted_label]

                print(f"Predicted Activity: {predicted_activity}")

                # p_class = predict_avg(input_path, model, output_size, num_frames, image_height, image_width, classes, webcam)
                result = [directory, predicted_activity, predicted_value]
                # all_results.append(result)

                # Get true labels and predictions
                # true_class = result[0]
                # pred = result[1]

                for r in result:
                    all_results.append({"true_class": r[0], 
                                "predicted_class": r[1], 
                                "predicted_value": r[2]})

                # print(result)

                sub_dir_count += 1

            sub_dir_count = 0
            dir_count += 1

        else:
            print("No video found")

    print("Done")

    return all_results




def predict_all_3D(test_path, model, num_frames, image_height, image_width, classes, output_size, webcam):
    # train_path = f'{path}/downloads/selected_features'
    # test_path = f'{path}/downloads/ignored_features'
    train_directory = list_files(test_path)

    all_results = [] 

    dir_len = len(train_directory)
    sub_dir_len = 0
    dir_count = 1
    sub_dir_count = 1

    for directory in train_directory:

        vid_path = f'{test_path}/{directory}'

        dir_files = list_files(vid_path)
        sub_dir_len = len(dir_files)

        print(" ")
        print("=========================================")
        print(f'Class: {directory}')
        print("=========================================")

        if sub_dir_len != 0:
            for video in dir_files:
                print(" ")
                print(f'Folder: {dir_count}/{dir_len}  |  File: {sub_dir_count}/{sub_dir_len}')
                print(" ")

                input_path = f'{vid_path}/{video}'

                # Make avg prediction for each video
                p_class = predict_avg(input_path, model, output_size, num_frames, image_height, image_width, classes, webcam)
                result = [directory, p_class]
                # all_results.append(result)

                # Get true labels and predictions
                true_class = result[0]
                pred = result[1]

                for p in pred:
                    all_results.append({"true_class": true_class, 
                                "predicted_class": p[0], 
                                "predicted_value": p[1]})

                # print(result)

                sub_dir_count += 1

            sub_dir_count = 0
            dir_count += 1

        else:
            print("No video found")

    print("Done")

    return 


# Function to load saved model and evaluate on new test data
def load_and_evaluate_model(model_path, test_data_folder, img_size=(64, 64), sequence_length=30, plot=False):
    # Load the trained model
    model = load_model(model_path)
    print(" ")
    print(f'Loaded model from: {model_path}')

    # Load and preprocess the test data
    print(" ")
    print('Preprocessing')

    test_data, test_labels, activity_classes = load_videos_from_folders(test_data_folder, img_size, sequence_length)

    # Make predictions
    print(" ")
    print('Making predictions')
    print(" ")

    test_predictions = model.predict(test_data)

    # Convert predictions and labels to class indices
    predicted_labels = np.argmax(test_predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(" ")
    print("=========================================")
    print(f"Test Accuracy: {accuracy}")
    print("=========================================")
    print(" ")

    if plot:
        # Generate confusion matrix
        # cmap = plt.cm.Blues
        cm = plot_confusion(t_class = true_labels, 
                            p_class = predicted_labels,
                            display_labels = activity_classes, 
                            title = "Conv3D Confusion")
        
        # Make classification report
        report = classification_report(true_labels, predicted_labels)
        print(report)

        return accuracy, predicted_labels, true_labels, cm

    else:
        return accuracy, predicted_labels, true_labels


if __name__ == "__main__":
    setup()
    list_files()
    get_classes()
    plot_classes()
    pre_process()
    frames_extraction()
    create_dataset()
    make_features_labels()
    create_model()
    train_model()
    predict_avg()
    predict_avg_stream()
    predict_all()
    plot_confusion()
    get_data()
    make_matrix_csv()
    load_videos_from_folders()
    video_to_frames()
    process_live_frames()
    predict_all_3D()
    load_and_evaluate_model()
