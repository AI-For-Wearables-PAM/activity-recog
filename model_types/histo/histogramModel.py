import cv2
import numpy as np
import os
import sys
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor

def extract_color_histogram(video_path, frame_sample_rate=30, bins=64):
    cap = cv2.VideoCapture(video_path)
    hist = np.zeros((bins, bins, bins))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_sample_rate == 0:
            # Convert frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for the frame
            hist_frame = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            
            # Accumulate histogram
            hist += hist_frame

        frame_count += 1

    cap.release()
    return hist.flatten()

def process_class_folder(class_folder, class_path, label_counter, bins, frame_sample_rate, tqdm=tqdm):
    data = []
    labels = []
    video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc=f"Processing videos in {class_folder}", leave=False):
        video_path = os.path.join(class_path, video_file)
        hist = extract_color_histogram(video_path, frame_sample_rate, bins)
        data.append(hist)
        labels.append(label_counter)
    
    return data, labels

def load_data(classes_folder, bins=64, frame_sample_rate=30, tqdm=tqdm):
    data = []
    labels = []
    label_map = {}
    label_counter = 0

    class_folders = [f for f in os.listdir(classes_folder) if os.path.isdir(os.path.join(classes_folder, f))]
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for class_folder in tqdm(class_folders, desc="Processing classes"):
            class_path = os.path.join(classes_folder, class_folder)
            label_map[label_counter] = class_folder
            futures.append(executor.submit(process_class_folder, class_folder, class_path, label_counter, bins, frame_sample_rate, tqdm=tqdm))
            label_counter += 1
        
        for future in tqdm(futures, desc="Collecting results"):
            class_data, class_labels = future.result()
            data.extend(class_data)
            labels.extend(class_labels)

    return np.array(data), np.array(labels), label_map

def train_model(data, labels, tqdm=tqdm):
    model = SVC(kernel='linear', verbose=True)
    model.fit(data, labels)
    
    return model

def process_single_video(args):
    video_path, class_folder, model, label_map, reverse_label_map = args
    hist = extract_color_histogram(video_path)
    prediction = model.predict([hist])[0]
    predicted_class = label_map[prediction]
    
    print()
    if predicted_class == class_folder:
        print("Accurate! ✅", predicted_class, class_folder)
    else:
        print("Inaccurate! ❌", predicted_class, class_folder)
        
    return (reverse_label_map[class_folder], prediction)

def classify_videos(validation_folder, model, label_map, tqdm=tqdm, max_workers=4):
    # Load the trained model and label map
    reverse_label_map = {v: k for k, v in label_map.items()}

    true_labels = []
    predicted_labels = []
    
    # Collect all video paths and their corresponding class folders
    video_tasks = []
    class_folders = [f for f in os.listdir(validation_folder) 
                    if os.path.isdir(os.path.join(validation_folder, f))]
    
    for class_folder in tqdm(class_folders, desc="Collecting video paths"):
        if class_folder not in reverse_label_map:   #For when the training data has videos that the testing data doesn't.
            print("Missing class? ❌", class_folder)
            continue    #This might throw off the accuracy
        class_path = os.path.join(validation_folder, class_folder)
        video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            # Create a tuple of arguments needed for processing
            video_tasks.append((video_path, class_folder, model, label_map, reverse_label_map))
    
    # Process videos in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress
        results = list(tqdm(
            executor.map(process_single_video, video_tasks),
            total=len(video_tasks),
            desc="Processing videos"
        ))
        
        # Unzip the results
        true_labels, predicted_labels = zip(*results)
    
    return true_labels, predicted_labels

if __name__ == "__main__":
    if len(sys.argv) not in [2, 4]:
        print("Usage: python script.py <path_to_classes_folder> [<path_to_model> <path_to_label_map>]")
        sys.exit(1)

    classes_folder = sys.argv[1]

    if len(sys.argv) == 4:
        model_path = sys.argv[2]
        label_map_path = sys.argv[3]
        model = joblib.load(model_path)
        label_map = joblib.load(label_map_path)

        # Classify videos and get true and predicted labels
        true_labels, predicted_labels = classify_videos(classes_folder, model, label_map)
        
        # Calculate and print accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Classification accuracy: {accuracy * 100:.2f}%")
    
    else:
        # Load data and train the model
        data, labels, label_map = load_data(classes_folder)
        model = train_model(data, labels)
    
        y_pred = model.predict(data)
        accuracy = accuracy_score(labels, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        # Save the trained model and label map to files
        joblib.dump(model, 'video_classification_model.pkl')
        joblib.dump(label_map, 'label_map.pkl')

        print("Model trained and saved as 'video_classification_model.pkl'.")
        print("Label map saved as 'label_map.pkl'.")
