# Patient Activity Monitoring  
This repo features our initial work using OpenCV and TensorFlow for human activity recognition. It modifies base code provided from the tutorial [Introduction to Video Classification & Human Activity Recognition](https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/). 

Two convolutional neural networks (CNNs) are trained using OpenCV and TensorFlow: a [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) and [Conv3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D) version. The Conv2D and Conv3D models contain 2 and 3 convolutional layers, respectively.

## Set up environment

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

## Conv2D Version

### Hyperparameters

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


## Conv3D Version
[...]

#### Architecture hyperparameters

[...]

#### Training hyperparameters

[...]

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

## movinet.py
If packages can't be installed using `pip` try `python3 -m pip install <package name>`

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
