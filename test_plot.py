from functions import make_matrix_csv
import matplotlib.pyplot as plt

# Define params
data_dir = './data'
csv_name = '2024-09-21-20-07-22-test.csv'
file_path = f'{data_dir}/{csv_name}'
title ='Conv2D Confusion'

# Plot results
all_predictions = make_matrix_csv(file_path = file_path,
                                  show_top = True,
                                  plot_title = title)

# Display interactive plot
plt.show() 