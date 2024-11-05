from conv3d_functions import load_and_evaluate_model
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Load model
model_path = './2024-11-04-22-47-17-conv3d-model.keras'

# Change to True for the option to change datasets 
choose_dataset = False

if choose_dataset:

    print(" ")
    print("Use test or train dataset?")
    print('----> Options: test | train | q')

    selection = input()

else:
    selection = "r2_test"

run = True
show_plot = True

# Predict each video
while run: 
    if selection == "r2_test" or selection == "train":
        subset = selection
        data_path = f'../downloads/{subset}'

        # Make predictions
        all_predictions = load_and_evaluate_model(model_path = model_path, 
                                                  test_data_folder = data_path)

        # Make DataFrame
        results_df = pd.DataFrame(all_predictions)

        # Timestamp
        timestamp = dt.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        # Save the DataFrame
        print("Saving data...")
        results_df.to_csv(f'../data/{timestamp}-{subset}.csv', index=False)

        if show_plot:
            plt.show()

        run = False

    elif selection == "q":
        run = False

    else: 
        print(" ")
        print('Invalid response. Enter "test" or "train". Enter "q" to quit.')

print("Done")  
