import cv2
import numpy as np
from functions import predict_avg, predict_avg_stream
from keras.models import load_model
from collections import deque
from tabulate import tabulate


def makePredTable(predictions):
    # Generate table of predictions
    table = []

    for p in predictions:   
        table.append([p[0], p[1]])

    print('')
    print('Predictions for each class:')
    print('')

    print(tabulate(table, 
                headers=["Prediction", "Probability"],
                tablefmt="outline"))

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
                'Asleep Trying to sleep','Family', 'Sitting In Wheelchair', 
                'Talking on the Phone']

img_height, img_width = 64, 64
max_images_per_class = 8000
frames = 12
window_size = 1
size = len(classes_list)

# Saves new clip when streaming predictions 
output_path = './predictions/new_capture.mp4'

run = True

while run:
    # Choose file or webcam
    print('')
    print('Predict from video file or webcam?')
    print('[1] Video file  [2] Webcam')
    format = input()

    if format == str(1):
        webcam = False
        test_video = './downloads/test/Nurse Visit/7395736338672438417.mp4'
    elif format == str(2):
        webcam = True
    else:
        webcam = False

    print('')
    print("Loading Conv2D")
    model_path = './models/2024-09-20-04-00-06-model.keras'

    # Demo needs to be adjusted for Conv3D
    # print('')
    # print("Loading Conv3D")
    # model_path = './conv3D/2024-09-22-16-02-44-conv3d-model.keras'

    print('')
    model = load_model(model_path)
    print(f'Loaded model from: {model_path}')
    print('')

    if webcam:
        # Stream rolling average of predictions over webcam video 
        print('')
        print('Stream predictions? y/n')
        stream = input()
        
        if stream == 'y':
            print('')
            print('Streaming predicitons. Press "q" to end stream.')
            predict_avg_stream(model = model, path = output_path, 
                               classes = classes_list, window = window_size, 
                               image_height = img_height, image_width = img_width, 
                               print_pred = True)
            
            # Ends program so OpenCV doesn't hang
            run = False

        # Briefly open webcam and make an average prediction
        elif stream == 'n':
            print('')
            print('Making predictions...')
            print('')
            predictions = predict_avg(directory = output_path,
                                    model = model, 
                                    output_size = size, 
                                    num_frames = frames, 
                                    image_height = img_height, 
                                    image_width = img_width, 
                                    classes = classes_list, 
                                    webcam = True)
            
            makePredTable(predictions)

            print('')
            run = rerun()
        
    elif webcam == False:
        # Predict from video file
        print('')
        print(f'Making predictions on {test_video}')
        print('')
        predictions = predict_avg(directory = test_video,
                                model = model, 
                                output_size = size, 
                                num_frames = frames, 
                                image_height = img_height, 
                                image_width = img_width, 
                                classes = classes_list, 
                                webcam = False)


        makePredTable(predictions)

        print('')
        run = rerun()

    else:
        print('Something went wrong. Check the model path.')

        print('')
        run = False