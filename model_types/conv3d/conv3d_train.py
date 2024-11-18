from conv3d_functions import train3d
import os

TF_ENABLE_ONEDNN_OPTS=1

param_grid = {'conv_filters': [32, 64, 128], 
              'kernel_size': (3, 3, 3), 
              'dense_units': 512, 
              'dropout_rate': 0.4, 
              'learning_rate': 0.001, 
              'batch_size': 8, 
              'epochs': 30}

# Path to training data
path = '../../downloads'
train_dir = f'{path}/fr_10s/train_fr_10s'

# Train model
best_model_details = train3d(train_dir, param_grid, random_search=False, complex=True)

for detail in best_model_details:
    print(detail)