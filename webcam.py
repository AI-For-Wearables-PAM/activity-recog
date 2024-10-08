import cv2
import numpy as np
from functions import predict_avg, predict_avg_stream
from keras.models import load_model
from collections import deque
from tabulate import tabulate

# Load model
# Conv2D
model_path = './models/2024-09-20-04-00-06-model.keras'
# Conv3D
model_path = './conv3D/2024-09-22-13-18-18-conv3d-model.keras'

model = load_model(model_path)
print(f'Loaded model from: {model_path}')

classes_list = ['Transfer To Bed',
                'Doctor Visit',
                'Nurse Visit',
                'Therapy',
                'EVS Visit',
                'Eating',
                'Lying In Bed',
                'Watching TV',
                'Asleep Trying to sleep',
                'Family',
                'Sitting In Wheelchair',
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
    # Stream rolling average of predictions over webcam video 
    print(' ')
    print('Stream predictions? y/n')
    stream = input()

    user_print = False
    valid_selection = False
    

    if stream == 'y':
            print("Print predicitons to the console? y/n")
            user_print = input()
            
            if user_print == "y":
                user_print = True
                valid_selection = True
            elif user_print == "n":
                user_print = False
                valid_selection = True
            else: 
                print('Invalid selection. Press "y" or "n"')

            if valid_selection:
                print('Streaming predicitons. Press "q" to end stream.')
                predict_avg_stream(model = model, 
                                path = output_path, 
                                classes = classes_list, 
                                window = window_size, 
                                image_height = img_height, 
                                image_width = img_width,
                                print_pred = user_print)
                
                print(' ')
                run = False

    # Briefly open webcam and make an average prediction
    elif stream == 'n':
        print('Making predictions...')
        predictions = predict_avg(directory = output_path,
                                  model = model, 
                                  output_size = size, 
                                  num_frames = frames, 
                                  image_height = img_height, 
                                  image_width = img_width, 
                                  classes = classes_list, 
                                  webcam = True)


        # Generate table of predictions
        table = []

        for p in predictions:   
            table.append([p[0], p[1]])
    
        print(' ')
        print('Predictions for each class:')
        print(' ')

        print(tabulate(table, 
                    headers=["Prediction", "Probability"],
                    tablefmt="outline"))

        print(' ')
        run = False

    else:
        print('Something went wrong. Check the model path.')

        print(' ')
        run = False
