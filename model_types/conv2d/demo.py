import cv2
import numpy as np
from conv2d.conv2d_functions import predict_avg, predict_avg_stream, list_files
from keras.models import load_model
from collections import deque
from tabulate import tabulate
import random

 
# Generate table of predictions
def makePredTable(predictions, true_class, file):
    table = []

    for p in predictions:   
        table.append([p[0], p[1]])

    print('')
    print(f'File: {file}')
    print(f'True class: {true_class}')
    
    print('')
    print('Predicted classes:')

    print(tabulate(table, 
                headers=["Prediction", "Probability"],
                tablefmt="outline"))

# Ask to run again
def rerun():
    print('')
    print('Run again? y/n')
    rerun = input()

    if rerun == 'n':
        print('Done')
        run = False
    else:
        run = True

    return run


classes_list = ['Transfer To Bed', 'Doctor Visit', 'Nurse Visit', 'Therapy',
                'EVS Visit', 'Eating', 'Lying In Bed', 'Watching TV', 
                'Asleep-Trying to sleep','Family', 'Sitting In Wheelchair', 
                'Talking on the Phone']

# Input configuration
img_height, img_width = 64, 64
max_images_per_class = 8000
frames = 12
window_size = 1
size = len(classes_list)

# Save new clip when streaming predictions (not working currently)
output_path = './predictions/new_capture.mp4'

# Using Conv2D. Demo needs to be adjusted for Conv3D.
print('')
print("Loading Conv2D")
model_path = './models/2024-09-20-04-00-06-model.keras'
model = load_model(model_path)
print(f'Loaded model from: {model_path}')

run = True

while run:
    print('')
    print('Predict from video file or webcam?')
    print('[1] Video file  [2] Webcam')
    format = input()

    # Choose file or webcam
    if format == str(1):
        webcam = False
        # test_video = './downloads/test/Nurse Visit/7395736338672438417.mp4'
    elif format == str(2):
        webcam = True
    else:
        webcam = False

    if webcam: 
        print('')
        print('Stream predictions? y/n')
        stream = input()
        
        if stream == 'y':
            print('')
            print('Streaming predicitons. Press "q" to end stream.')

            # Stream rolling average of predictions over webcam video
            predict_avg_stream(model = model, path = output_path, 
                               classes = classes_list, window = window_size, 
                               image_height = img_height, image_width = img_width, 
                               print_pred = True)
            
            # Ends program so OpenCV doesn't hang
            run = False

        elif stream == 'n':
            print('')
            print('Making predictions...')
            print('')

            # Briefly open webcam and make an average prediction
            predictions = predict_avg(directory = output_path,
                                    model = model, 
                                    output_size = size, 
                                    num_frames = frames, 
                                    image_height = img_height, 
                                    image_width = img_width, 
                                    classes = classes_list, 
                                    webcam = True)
            
            makePredTable(predictions, true_class="", file="")

            print('')
            run = rerun()
        
    elif webcam == False:
        # Select one of the 12 classes
        print('')
        print(f'Select a class:')

        class_num = 0

        for c in classes_list:
            print(f'[{class_num}] {c}')
            class_num += 1

        selected_class = int(input())
        selected_class = classes_list[selected_class]

        test_dir = f'./downloads/test/{selected_class}'
        files = list_files(test_dir)

        # Select random video
        random_video = random.choice(files)
        random_path = f'{test_dir}/{random_video}'
        
        print('')
        print(f'Predicting {random_video}')

        # Predict random video file
        predictions = predict_avg(directory = random_path,
                                  model = model, 
                                  output_size = size, 
                                  num_frames = frames, 
                                  image_height = img_height, 
                                  image_width = img_width, 
                                  classes = classes_list, 
                                  webcam = False)

        makePredTable(predictions, 
                      true_class = selected_class,
                      file = random_video)

        print('')
        run = rerun()

    else:
        print('Something went wrong. Check the model path.')
        print('')
        run = False