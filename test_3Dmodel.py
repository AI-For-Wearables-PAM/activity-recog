from functions import load_videos_from_folders, predict_all, predict_3D
from keras.models import load_model
import pandas as pd
import datetime as dt

# Load model
model_path = './conv3D/2024-09-22-16-02-44-conv3d-model.keras'
model = load_model(model_path)
print(f'Loaded model from: {model_path}')

# Predict each video
print(" ")
print("Use test or train dataset?")
print('----> Options: test | train | q')

selection = input()

run = True

while run: 
    if selection == "test" or selection == "train":
        subset = selection
        path = f'./downloads/{subset}'

        # Get class names
        pre_pro = load_videos_from_folders(path)

        classes_list = pre_pro[1]
        model_output_size = pre_pro[1][2]

        # Define params
        img_height, img_width = 64, 64
        max_images_per_class = 8000
        frames = 12
        window_size = 1
        size = len(classes_list)

        # Make predictions
        all_predictions = predict_3D(test_path = path,
                                     model = model,
                                     activity_classes = classes_list)

        # Make DataFrame
        results_df = pd.DataFrame(all_predictions)

        # Timestamp
        timestamp = dt.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        # Save the DataFrame
        print("Saving data")
        results_df.to_csv(f'./data/{timestamp}-{subset}.csv', index=False)

        run = False

    elif selection == "q":
        run = False

    else: 
        print(" ")
        print('Invalid response. Enter "test" or "train". Enter "q" to quit.')

