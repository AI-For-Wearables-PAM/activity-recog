from functions import load_and_evaluate_model
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Load model
model_path = './conv3D/2024-09-22-16-02-44-conv3d-model.keras'

# Change to True for the option to change datasets 
choose_dataset = False

if choose_dataset:

    print(" ")
    print("Use test or train dataset?")
    print('----> Options: test | train | q')

    selection = input()

else:
    selection = "test"

run = True
show_plot = False

# Predict each video
while run: 
    if selection == "test" or selection == "train":
        subset = selection
        data_path = f'./downloads/{subset}'

        # Make predictions
        all_predictions = load_and_evaluate_model(model_path = model_path, 
                                                  test_data_folder = data_path,
                                                  plot = show_plot)

        # # Make DataFrame
        # results_df = pd.DataFrame(all_predictions)

        # # Timestamp
        # timestamp = dt.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        # # Save the DataFrame
        # print("Saving data...")
        # results_df.to_csv(f'./data/{timestamp}-{subset}.csv', index=False)

        # if show_plot:
        #     plt.show()

        run = False

    elif selection == "q":
        run = False

    else: 
        print(" ")
        print('Invalid response')

print("Done")