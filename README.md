# Patient Activity Monitoring  
This repo features our initial work using OpenCV, TensorFlow, and PyTorch to train three convolutional neural networks (CNNs) for human activity recognition. The code here includes modifications of the base code provided by the [Introduction to Video Classification & Human Activity Recognition](https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/) tutorial ([Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)), modifications to a pretrained model ([s3d](https://pytorch.org/vision/main/models/generated/torchvision.models.video.s3d.html#torchvision.models.video.s3d)), and a ([Conv3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D)) model built from scratch by @nehabaddam. 

Two convolutional neural networks (CNNs) are trained using OpenCV and TensorFlow: a [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [Conv3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D) version. The Conv2D and Conv3D models contain 2 and 3 convolutional layers, respectively.

## Setup local environment

### Create a virtual environment 
Use `venv` or `conda` and install the dependencies. 

### If using Apple Silicon GPU
The `tensorflow-metal` plugin will enable the GPU on Macs fitted with [Apple silicon](https://support.apple.com/en-us/116943) or AMD procesors, which radically improves model training time. More info is available [here](https://pypi.org/project/tensorflow-metal/). 


#### 1. Set up environment 
**Deactivate current venv / conda environments**
venv
```console
deactivate
```

conda
```console
conda deactivate
```

**Create new venv**
```console
python3 -m venv ~/venv-metal  
```  

**Activate venv-metal**
```console
source ~/venv-metal/bin/activate  
```  

#### 2. Install tensorflow-metal

```console
python -m pip install -U pip  
python -m pip install tensorflow-metal
```

#### 3. Install TensorFlow and OpenCV
```
python -m pip install tensorflow
pip3 install opencv-python
```

#### 4. Install dependencies from requirements.txt file (optional)

```console
pip install -r requirements.txt
```

#### 5. Check package versions 

```console
pip show <package name>
```

## File structure
The file `train_model.py` handles model training. Training and testing data is placed in the `downloads` folder at the root of your local clone of this repo. This directory is ignored for storage and PII reasons and will need to be added manually.

```
root
└───downloads  
│   └───test
│   │   └─feature
│   │     │   video1.mp4
│   │     │   video2.mp4
│   │     │   ...
│   │
│   └───train
│   │   └─feature
│   │     │   video3.mp4
│   │     │   video4.mp4
│   │     │   ...
```


#### Architecture hyperparameters

[...]

#### Training hyperparameters

[...]


### Train model

```
trained_model = trainModel(...)
```

### Load model

```
model = load_model(model_path)
```

### Inference

#### Short duration sample from live video
```
predictions = predict_avg(...)
```
Example output: 
```
Predictions for each class:
 
+------------------------+---------------+
| Prediction             |   Probability |
+========================+===============+
| EVS Visit              |             1 |
| Lying In Bed           |             0 |
| Family                 |             0 |
| Talking on the Phone   |             0 |
| Sitting In Wheelchair  |             0 |
| Asleep Trying to sleep |             0 |
| Watching TV            |             0 |
| Eating                 |             0 |
| Therapy                |             0 |
| Nurse Visit            |             0 |
| Doctor Visit           |             0 |
| Transfer To Bed        |             0 |
+------------------------+---------------+
```

#### Live video with media player
From `webcam.py`
```
window_size = 1
output_video_file_path = './predictions/new_capture.mp4'  
predict_on_live_video(output_video_file_path, window_size)
```


#### Architecture hyperparameters

[...]

#### Training hyperparameters

[...]

### Dependencies 
Dependencies can be installed manually or from `requirements.txt` using the command below. 

```console
pip install -r ./requirements.txt
```

### Training 
Ensure the videos organized in the format described in the _File structure_ section above. 

#### Train Conv2D

```console
python conv2d_train.py
```

#### Train Conv3D 

```console
python conv3d_train.py
```

#### Train s3d 

Navigate to `/notebooks/pre_trained.jpynb` and run the code in the Jupyter Notebook. 

### Testing 

#### Test Conv2D 
Ensure the videos organized in the format described in the _File structure_ section above, then run the command below and follow the prompts. 

```console
python conv2d_train.py
```

#### Test Conv3d 
This model is too large to store on GitHub, but you can download the model [here](https://drive.google.com/file/d/11Uh4Fwc-7eWNWMYk5FLqjpHCURRmY8zM/view?usp=drive_link) and place it in `./conv3D/2024-09-22-13-18-18-conv3d-model.keras`.

Next, ensure the videos organized in the format described in the _File structure_ section above, run the command below, and follow the prompts. 

```console
python conv3d_test.py 
```

#### Test s3d 
Ensure the videos organized in the format described in the _File structure_ section above. Navigate to `/notebooks/pre_trained.ipynb` and run the code in the Jupyter Notebook.

# Using pretrained models

## Troubleshooting
The file `pre_trained.ipynb` uses OpenCV features with dependencies that may not be automatically installed (e.g., GStreamer). To clear errors, OpenCV may need to be built from source. A few options are below.

1. ### [Example 1: Docs from OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html#tutorial_linux_install_quick_build_contrib)
```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
 
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
 
# Create build directory and switch into it
mkdir -p build && cd build
 
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
 
# Build
cmake --build .

```

2. ### [Example 2: A different method for building OpenCV from source](https://discuss.bluerobotics.com/t/opencv-python-with-gstreamer-backend/8842)
```
git clone --recursive https://github.com/skvark/opencv-python.git
cd opencv-python
export CMAKE_ARGS="-DWITH_GSTREAMER=ON"
pip install --upgrade pip wheel
# this is the build step - the repo estimates it can take from 5 
#   mins to > 2 hrs depending on your computer hardware
pip wheel . --verbose
pip install opencv_python*.whl
```

### Install tf-models-no-deps
If you're unable to use `tf-models-official`, do this instead.
```console
pip install tf-models-no-deps
```

## References
- [Introduction to Video Classification & Human Activity Recognition](https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/)
- [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
- [Conv3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D)
- [Apple silicon](https://support.apple.com/en-us/116943)
- [tensorflow-metal](https://pypi.org/project/tensorflow-metal/)
- [movinet](https://www.tensorflow.org/hub/tutorials/movinet)
