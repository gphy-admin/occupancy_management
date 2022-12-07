# Mobilenet + SSD - TF Object Detection API for occupancy management

This repository contains the training and test code for infrared object detection for people counting on GAP Processors.


## Content

1. **Object Detection API**: This folders contains the training scripts for TF object detection API for people counting
2. **Evaluation**: Python notebook for results accuracy evaluation
3. **TFlite_test**: Test scripts for tflite converted model
4. **NNtool**: Test scripts for NNTool converted model


## Setup the Dataset


The dataset could be found at [this address](https://drive.google.com/file/d/13uxgy8y7DKHUoYHcAhF4eJmOUbaafdfz/view?usp=sharing).

To download with wget:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13uxgy8y7DKHUoYHcAhF4eJmOUbaafdfz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13uxgy8y7DKHUoYHcAhF4eJmOUbaafdfz" -O DataSetV2.zip && rm -rf /tmp/cookies.txt
```

Then unzip it:
```
unzip DataSetV2.zip -d .
```

Then you can use the following script to copy the annotations and images inside the correct folders:
**Before running the script make sure that PRJ_ROOT_DIR variable is correctly set**

```
./copy_files_to_project.sh
```

