from functions import predict_all, get_classes, pre_process
from keras.models import load_model
import pandas as pd
import datetime as dt

# Load model
model_path = './models/2024-09-20-05-11-02-model.keras'
model = load_model(model_path)
print(f'Loaded model from: {model_path}')

# Predict each video in test set
test_path = './downloads/test'

# Get class names
all_classes_names = get_classes(test_path)

pre_pro = pre_process(test_path)

classes_list = pre_pro[0]
model_output_size = pre_pro[1]

# Define params
img_height, img_width = 64, 64
max_images_per_class = 8000
frames = 12
window_size = 1
size = len(classes_list)

# Make predictions
all_predictions = predict_all(test_path = test_path,
                              model = model,
                              output_size = size,
                              num_frames = frames, 
                              image_height = img_height, 
                              image_width = img_width, 
                              classes = classes_list,
                              webcam = False)

# Make DataFrame
results_df = pd.DataFrame(all_predictions)

# Timestamp
timestamp = dt.datetime.now()
today = dt.datetime.date(timestamp)

# Save the DataFrame
print("Saving data")
results_df.to_csv(f'./data/{today}-all_predictions.csv', index=False)
