import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model_path = './Model___Date_Time_2024_09_12__06_45_44___Loss_0.42576849460601807___Accuracy_0.9131770730018616.h5'
model = load_model(model_path)
print(f"Loaded best model from: {model_path}")

image_height, image_width = 64, 64
max_images_per_class = 8000

classes_list = ['Brought Back From Therapy (Transfer To Bed)',
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

model_output_size = len(classes_list)

def predict_on_live_video(output_file_path, window_size):
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Open webcam
    video_reader = cv2.VideoCapture(0)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

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
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
        
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show video player
        cv2.imshow('Predicted Frames', frame)

        # Writing The Frame, this saves the video
        video_writer.write(frame)    

        print('Predicted class name', predicted_class_name)

        # Press 'q' to exit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the VideoCapture and VideoWriter objects and release all resources held by them. 

    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

window_size = 1
output_video_file_path = './predictions/new_capture.mp4'
predict_on_live_video(output_video_file_path, window_size)