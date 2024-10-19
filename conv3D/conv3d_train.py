from conv3d_functions import train3d

param_grid = {'conv_filters': [32, 64, 128], 
              'kernel_size': (3, 3, 3), 
              'dense_units': 512, 
              'dropout_rate': 0.4, 
              'learning_rate': 0.001, 
              'batch_size': 8, 
              'epochs': 10}

# Path to training data
path = '../downloads'
train_dir = f'{path}/test'

# Train model
best_params, best_accuracy, best_model_path = train3d(train_dir, param_grid, random_search=False)

print(best_params)
print(best_accuracy)
print(best_model_path)