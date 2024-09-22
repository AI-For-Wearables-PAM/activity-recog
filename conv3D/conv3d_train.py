from conv3d_functions import *

param_grid = {
    'conv_filters': [[32, 64, 128], [64, 128, 256]],  # Filter sizes for the 3 Conv layers
    'kernel_size': [(3, 3, 3), (5, 5, 5)],  # Kernel size for Conv layers
    'dense_units': [512, 1024],  # Number of units in Dense layer
    'dropout_rate': [0.4, 0.5],  # Dropout rates
    'learning_rate': [0.001, 0.0001],  # Learning rates
    'batch_size': [8, 16],  # Batch size
    'epochs': [10, 20]  # Number of epochs
}

# Path to training data
path = '../downloads'
train_dir = f'{path}/train'

# Run grid search over multiple hyperparameters
best_params, best_accuracy, best_model_path = run_random_search(train_dir, param_grid)

print(best_params)
print(best_accuracy)
print(best_model_path)