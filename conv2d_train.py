from functions import *
from quickstart import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Path to training data
path = './downloads'
train_dir = f'{path}/train'

# Seed model
seed_constant = 23
setup(seed_constant)

# Get class names
all_classes_names = get_classes(train_dir)

pre_pro = pre_process(train_dir)

classes_list = pre_pro[0]
model_output_size = pre_pro[1]

# Define image size
image_height, image_width = 64, 64
max_images_per_class = 8000

# Make dataset
dataset = make_features_labels(train_dir, 
                               classes_list, 
                               image_height, 
                               image_width, 
                               max_images_per_class)

features = dataset[0]
labels = dataset[1]

# Build model
model = create_model(image_height, 
                     image_width, 
                     model_output_size)

# Convert labels into one-hot-encoded vectors
one_hot_encoded_labels = to_categorical(labels)

# Train/test/split dataset
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            one_hot_encoded_labels,
                                                                            test_size = 0.2,
                                                                            shuffle = True,
                                                                            random_state = seed_constant)

# Early Stopping Callback
early_stopping_callback = EarlyStopping(monitor = 'val_loss', 
                                        patience = 15,
                                        mode = 'min', 
                                        restore_best_weights = True)


# Train model
trained_model = train_model(model, 
                           features_train, 
                           labels_train, 
                           features_test, 
                           labels_test, 
                           epochs=10)